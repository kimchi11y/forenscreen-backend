from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
import hashlib
from datetime import datetime
import uuid
from PIL import Image
from PIL import UnidentifiedImageError
import io
import imagehash
from pillow_heif import register_heif_opener
import mimetypes
from pathlib import Path
import json
from sqlalchemy import text

app = FastAPI()

# Evidence copies saved during a check. Anchored to this file rather than the
# working directory, so starting the server from the repo root and from
# backend/ no longer writes to two different folders.
#
# Deliberately NOT mounted as static files: every image is served through
# /evidence/{id}/image, which checks it against its stored SHA-256 first.
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

# Pillow format name -> extension for the stored copy. Anything Pillow decodes
# but is not listed here is still stored, under a neutral ".img".
_IMAGE_EXTENSIONS = {
    "JPEG": ".jpg",
    "PNG": ".png",
    "HEIF": ".heic",
    "WEBP": ".webp",
    "GIF": ".gif",
    "BMP": ".bmp",
    "TIFF": ".tiff",
}


def _extension_for(image_format: str | None) -> str:
    return _IMAGE_EXTENSIONS.get((image_format or "").upper(), ".img")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", "http://localhost:5174", "http://localhost:3000",
        # The frontend dev server runs over HTTPS (needed for the auth cookie).
        "https://localhost:5173", "https://localhost:5174",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"message": "Welcome to ForenScreen API"}


@app.get("/health")
def health_check():
    return {"status": "healthy"}


def analyze_screenshot_forensics(image_content: bytes) -> dict:
    """
    Basic facts about an uploaded image: size, format, and when it was checked.

    Error Level Analysis, colour-variance and compression heuristics used to run
    here. They were removed: ELA flags every sharp edge in a PNG screenshot and
    is blinded by the re-compression every social platform applies, and none of
    them ever fed the verdict. Whether an image was edited is decided by
    comparing it with its registered original (see matching.py).
    """
    forensic_data = {
        "image_analysis": {},
        "timeline_data": {},
    }

    try:
        image = Image.open(io.BytesIO(image_content))

        # Basic image info
        forensic_data["image_analysis"] = {
            "width": image.width,
            "height": image.height,
            "format": image.format,
            "mode": image.mode,
            "aspect_ratio": f"{image.width}/{image.height}",
            "megapixels": round((image.width * image.height) / 1000000, 2)
        }

        # File timeline data
        forensic_data["timeline_data"] = {
            "analysis_timestamp": datetime.now().isoformat(),
            "file_size_bytes": len(image_content),
            "file_size_mb": round(len(image_content) / (1024 * 1024), 2),
            "format": image.format
        }

    except Exception as e:
        forensic_data["analysis_error"] = str(e)

    return forensic_data


import hashlib
import io
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from PIL import Image
import imagehash

import auth
import blockchain
import custody
import matching
from database import SessionLocal, ensure_schema, get_db
from models import CustodyEntry, EvidenceCheck, OriginalAsset

# Pillow does not decode HEIC/HEIF files unless this plugin is registered.
register_heif_opener()

# Ensure tables exist. Tolerates an unreachable database so the process still
# starts and can serve /health instead of hanging on connect.
ensure_schema()

@app.post("/upload-asset/", status_code=status.HTTP_201_CREATED)
async def upload_original_asset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    user: dict | None = Depends(auth.optional_user),
    db: Session = Depends(get_db),
):
    # 1. Read the raw incoming file data into memory
    file_bytes = await file.read()
    
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # 2. Decode before opening a database transaction so bad input is a 400.
        image = Image.open(io.BytesIO(file_bytes))
        image.load()
        calculated_phash = str(imagehash.phash(image)) # Produces a 16-character hex string
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a supported image (PNG, JPEG, or HEIC/HEIF).",
        ) from exc

    # 3. Compute the strict cryptographic signature (SHA-256)
    sha256_sig = hashlib.sha256(file_bytes).hexdigest()

    # 4. Refuse to register bytes we already hold. Two byte-identical originals
    #    share a pHash exactly, so a second copy sits at distance 0 from the
    #    first; with an ambiguity margin of 4 bits that pushed every later
    #    verification of that file into AMBIGUOUS_MATCH instead of
    #    VERIFIED_AUTHENTIC. Returning the existing row keeps the registry 1:1
    #    with distinct files and makes re-registration idempotent.
    existing = (
        db.query(OriginalAsset)
        .filter(OriginalAsset.sha256_hash == sha256_sig)
        .order_by(OriginalAsset.created_at.asc())
        .first()
    )
    if existing is not None:
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Asset already registered - returning the existing record",
                "duplicate": True,
                "id": str(existing.id),
                "file_name": existing.file_name,
                "phash": existing.phash,
                "sha256": existing.sha256_hash,
            },
        )

    # 5. Bind information to our database record
    new_asset = OriginalAsset(
        file_name=file.filename,
        file_data=file_bytes, # Storing binary payload directly
        phash=calculated_phash,
        sha256_hash=sha256_sig
    )

    try:
        db.add(new_asset)
        db.flush()

        # Same transaction as the insert: an asset that exists without its
        # registration entry would be a hole in the custody record.
        custody.append_entry(
            db,
            subject_type=custody.SUBJECT_ASSET,
            subject_id=new_asset.id,
            event_type=custody.ASSET_REGISTERED,
            actor=auth.actor_from(user),
            payload={
                "sha256": sha256_sig,
                "phash": calculated_phash,
                "file_name": file.filename,
                "size_bytes": len(file_bytes),
                "actor_email": (user or {}).get("email"),
            },
        )

        db.commit()
        db.refresh(new_asset)

        # Anchoring runs after the response is returned. A Sepolia block takes
        # roughly 12 seconds and the network can be down; neither is a reason to
        # make someone wait to save evidence, or to fail their registration.
        background_tasks.add_task(_anchor_asset, str(new_asset.id), sha256_sig)
    except IntegrityError:
        # Lost a race against a concurrent registration of the same bytes. The
        # UNIQUE index on sha256_hash is the source of truth, so return theirs.
        db.rollback()
        winner = (
            db.query(OriginalAsset)
            .filter(OriginalAsset.sha256_hash == sha256_sig)
            .first()
        )
        if winner is None:
            raise
        return JSONResponse(
            status_code=status.HTTP_200_OK,
            content={
                "message": "Asset already registered - returning the existing record",
                "duplicate": True,
                "id": str(winner.id),
                "file_name": winner.file_name,
                "phash": winner.phash,
                "sha256": winner.sha256_hash,
            },
        )
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process image asset: {str(e)}")

    return {
        "message": "Asset safely stored in Neon DB",
        "duplicate": False,
        "id": str(new_asset.id),
        "file_name": new_asset.file_name,
        "phash": new_asset.phash,
        "sha256": new_asset.sha256_hash
    }


@app.get("/view-asset/{asset_id}")
def get_asset_image(asset_id: str, db: Session = Depends(get_db)):
    # Helper route so your React frontend can visually load the binary image string
    asset = db.query(OriginalAsset).filter(OriginalAsset.id == asset_id).first()
    if not asset:
        raise HTTPException(status_code=404, detail="Asset metadata reference not found.")
    
    # Stream binary data directly back out as a standard displayable JPEG/PNG
    return StreamingResponse(io.BytesIO(asset.file_data), media_type="image/jpeg")

# pHash is a pre-filter, not a verdict. It is kept deliberately loose because
# its only job is to pick which registered originals are worth a full ORB
# comparison - a 25% crop alone moves pHash by ~24 bits, and rejecting a
# candidate here means it is never offered to the matcher.
PHASH_PREFILTER_THRESHOLD = 24
PHASH_PREFILTER_LIMIT = 5

# Still used as the fallback signal when ORB cannot judge a candidate at all
# (texture-poor image), where a strict distance is the right thing to demand.
PHASH_MATCH_THRESHOLD = 10

# Two candidates whose inlier counts are within this fraction of each other are
# geometrically indistinguishable, so the verdict must not silently pick one.
ORB_AMBIGUITY_INLIER_RATIO = 0.15

# ORB reasons that mean "could not form an opinion", as opposed to "looked and
# decided these are different pictures". Only the former may fall back to pHash.
ORB_UNVERIFIABLE = {
    "too_few_keypoints",
    "too_few_good_matches",
    "homography_failed",
    "undecodable_original",
}


@app.post("/upload")
async def upload_evidence(
    file: UploadFile = File(...),
    case_id: str = Form(None),
    submitter_metadata: str = Form(None),
    # Optional explicit scoping: if the caller already knows which registered
    # original this evidence should be verified against (e.g. selected from
    # a case's asset list in the UI), pass its id here to bypass the
    # database-wide nearest-neighbor search entirely. This is the most
    # reliable fix for ambiguous pHash matches - prefer it when available.
    original_asset_id: str = Form(None),
    user: dict | None = Depends(auth.optional_user),
    db: Session = Depends(get_db)
):
    """
    Upload evidence files and generate forensic analysis for cyberbullying cases.
    Specialized for screenshot forensics: visual similarity matching, edit detection,
    and cryptographic integrity checking using Approach A.
    """
    results = []

    # Parse submitter metadata if provided
    metadata = {}
    if submitter_metadata:
        try:
            metadata = json.loads(submitter_metadata)
        except json.JSONDecodeError:
            print("Invalid JSON in submitter_metadata")

    # Process single file
    files = [file]  # Wrap in list for consistent processing

    for file in files:
        # Read file content into memory
        file_content = await file.read()
        if not file_content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        # Reject anything that is not an image before doing any work on it. The
        # browser's content type is only a claim, so the bytes are decoded.
        try:
            image = Image.open(io.BytesIO(file_content))
            image.load()
        except (UnidentifiedImageError, OSError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a supported image (PNG, JPEG, or HEIC/HEIF).",
            ) from exc

        # Calculate SHA-256 hash for strict file integrity matching
        sha256_hash = hashlib.sha256(file_content).hexdigest()
        file_id = str(uuid.uuid4())

        # The extension comes from the decoded format, never from the client's
        # filename, so a file named "x.html" cannot be stored as one.
        saved_filename = f"{file_id}{_extension_for(image.format)}"

        # 1. EXACT FILE MATCH FIRST. If these exact bytes are already registered
        #    the answer is definitive and needs no perceptual reasoning: the file
        #    is unmodified. Doing this before the pHash search is what stops a
        #    byte-identical re-upload from coming back as AMBIGUOUS_MATCH whenever
        #    the registry holds more than one visually similar original - which
        #    includes the same image having been registered more than once.
        exact_hit = db.execute(
            text("""
                SELECT id, file_name, phash, sha256_hash
                FROM original_assets
                WHERE sha256_hash = :sha256
                ORDER BY created_at ASC
                LIMIT 1;
            """),
            {"sha256": sha256_hash},
        ).fetchone()

        # Set inside the image branch; stays None for files we cannot decode.
        computed_phash = None

        if exact_hit is not None:
            verification_status = "VERIFIED_AUTHENTIC"
            matched_asset_info = {
                "original_asset_id": str(exact_hit.id),
                "original_filename": exact_hit.file_name,
                "original_sha256": exact_hit.sha256_hash,
                "match_method": "sha256_exact",
            }
        else:
            verification_status = "UNKNOWN_ASSET"
            matched_asset_info = None

        forensic_data = {}

        try:
            # 2. Compute the Visual Perceptual Hash (pHash) in memory
            computed_phash = str(imagehash.phash(image))

            # 2. pHash pre-filter. It only decides which registered originals
            #    are worth the cost of a full comparison, so it is deliberately
            #    generous - a 25% crop moves pHash by ~24 bits. ORB, not this
            #    threshold, is what rejects a bad candidate.
            if exact_hit is None:
                if original_asset_id:
                    # Explicit scoping: the caller already knows which
                    # original this evidence belongs to.
                    candidates = db.execute(
                        text("""
                            SELECT id, file_name, phash, sha256_hash, 0 AS distance
                            FROM original_assets
                            WHERE id = :asset_id;
                        """),
                        {"asset_id": original_asset_id},
                    ).fetchall()
                else:
                    candidates = db.execute(
                        text("""
                            SELECT id, file_name, phash, sha256_hash,
                                   BIT_COUNT(('x' || phash)::bit(64) # ('x' || :query_phash)::bit(64)) AS distance
                            FROM original_assets
                            WHERE BIT_COUNT(('x' || phash)::bit(64) # ('x' || :query_phash)::bit(64)) <= :threshold
                            ORDER BY distance ASC
                            LIMIT :limit;
                        """),
                        {
                            "query_phash": computed_phash,
                            "threshold": PHASH_PREFILTER_THRESHOLD,
                            "limit": PHASH_PREFILTER_LIMIT,
                        },
                    ).fetchall()

                # 3. Load the candidate bytes. Fetched here rather than in the
                #    search above so we do not pull megabytes of BYTEA for
                #    candidates that ORB is never going to look at.
                payload = []
                for candidate in candidates:
                    blob = db.execute(
                        text("SELECT file_data FROM original_assets WHERE id = :id"),
                        {"id": candidate.id},
                    ).fetchone()
                    if blob is None or blob.file_data is None:
                        continue
                    payload.append({
                        "id": candidate.id,
                        "file_name": candidate.file_name,
                        "sha256_hash": candidate.sha256_hash,
                        "phash_distance": int(candidate.distance),
                        "file_data": bytes(blob.file_data),
                    })

                # 4. ORB + RANSAC decides. CPU-bound, so it goes to a worker
                #    thread instead of blocking the event loop.
                ranked = (
                    await run_in_threadpool(
                        matching.rank_candidates, file_content, payload
                    )
                    if payload
                    else []
                )

                winners = [r for r in ranked if r["ok"]]

                if (
                    len(winners) >= 2
                    and (winners[0]["inliers"] - winners[1]["inliers"])
                    <= ORB_AMBIGUITY_INLIER_RATIO * winners[0]["inliers"]
                ):
                    verification_status = "AMBIGUOUS_MATCH"
                    matched_asset_info = {
                        "candidates": [
                            {
                                "original_asset_id": str(w["original_asset_id"]),
                                "original_filename": w["original_filename"],
                                "original_sha256": w["original_sha256"],
                                "orb_inliers": w["inliers"],
                                "phash_distance": w["phash_distance"],
                            }
                            for w in winners
                        ],
                        "message": (
                            "More than one registered original is geometrically "
                            "consistent with this evidence. Re-verify against a "
                            "specific original_asset_id to resolve."
                        ),
                    }
                elif winners:
                    best = winners[0]
                    verification_status = "TAMPERED_MANIPULATED"
                    matched_asset_info = {
                        "original_asset_id": str(best["original_asset_id"]),
                        "original_filename": best["original_filename"],
                        "original_sha256": best["original_sha256"],
                        "match_method": "orb_ransac",
                        "orb_inliers": best["inliers"],
                        "orb_inlier_ratio": best["inlier_ratio"],
                        "orb_good_matches": best["good_matches"],
                        "phash_distance": best["phash_distance"],
                    }
                elif ranked and all(
                    r["reason"] in ORB_UNVERIFIABLE for r in ranked
                ):
                    # ORB could not judge any candidate (texture-poor or very
                    # flat image). Fall back to the pHash distance as a weaker
                    # signal rather than silently calling the evidence
                    # unregistered, and say so in match_method.
                    fallback = min(
                        (r for r in ranked if r["phash_distance"] is not None),
                        key=lambda r: r["phash_distance"],
                        default=None,
                    )
                    if (
                        fallback is not None
                        and fallback["phash_distance"] <= PHASH_MATCH_THRESHOLD
                    ):
                        verification_status = "TAMPERED_MANIPULATED"
                        matched_asset_info = {
                            "original_asset_id": str(fallback["original_asset_id"]),
                            "original_filename": fallback["original_filename"],
                            "original_sha256": fallback["original_sha256"],
                            "match_method": "phash_fallback",
                            "phash_distance": fallback["phash_distance"],
                        }
                    else:
                        verification_status = "UNREGISTERED_ASSET"
                else:
                    verification_status = "UNREGISTERED_ASSET"

            forensic_data = analyze_screenshot_forensics(file_content)

        except Exception as img_err:
            print(f"Error processing visual forensics for {file.filename}: {img_err}")
            verification_status = "PROCESSING_ERROR"

        # Create comprehensive result payload for each file
        file_result = {
            "file_id": file_id,
            "filename": file.filename,
            # Filled in only once the check is stored: an unstored check keeps
            # no copy of the file, so there is nothing to point at.
            "saved_filename": None,
            "size": len(file_content),
            "mime_type": file.content_type,
            "sha256_hash": sha256_hash,
            "case_id": case_id,
            "uploaded_at": datetime.now().isoformat(),
            "verification_status": verification_status,  # AUTHENTIC, TAMPERED, or UNREGISTERED
            "matched_asset": matched_asset_info,          # Details of the original file it matched with
            "forensics": forensic_data,
            "submitter_metadata": metadata  
        }

        # Persist the check and record it in the chain of custody. A verdict that
        # is not stored cannot be revisited, exported or audited - and you cannot
        # log the handling of a record you do not keep.
        #
        # A storage failure must not invalidate a verdict the user has already
        # been given, so it is reported in the response rather than raised.
        #
        # The evidence copy is written to disk inside the same try as the
        # commit, and removed again if the commit fails, so the uploads folder
        # never holds a file the database does not know about.
        file_result["persisted"] = False
        if computed_phash is not None:
            file_path = UPLOAD_DIR / saved_filename
            matched_asset_id = None
            if isinstance(matched_asset_info, dict) and "candidates" not in matched_asset_info:
                matched_asset_id = matched_asset_info.get("original_asset_id")

            try:
                db.add(
                    EvidenceCheck(
                        id=uuid.UUID(file_id),
                        asset_id=uuid.UUID(matched_asset_id) if matched_asset_id else None,
                        sha256_hash=sha256_hash,
                        phash=computed_phash,
                        verdict=verification_status,
                        match_method=(matched_asset_info or {}).get("match_method"),
                        orb_inliers=(matched_asset_info or {}).get("orb_inliers"),
                        orb_inlier_ratio=(matched_asset_info or {}).get("orb_inlier_ratio"),
                        phash_distance=(matched_asset_info or {}).get("phash_distance"),
                        saved_filename=saved_filename,
                    )
                )
                custody.append_entry(
                    db,
                    subject_type=custody.SUBJECT_EVIDENCE,
                    subject_id=uuid.UUID(file_id),
                    event_type=custody.CHECK_PERFORMED,
                    actor=auth.actor_from(user),
                    payload={
                        "sha256": sha256_hash,
                        "verdict": verification_status,
                        "match_method": (matched_asset_info or {}).get("match_method"),
                        "orb_inliers": (matched_asset_info or {}).get("orb_inliers"),
                        "matched_asset_id": matched_asset_id,
                        "saved_filename": saved_filename,
                    },
                )
                file_path.write_bytes(file_content)
                db.commit()
                file_result["persisted"] = True
                file_result["saved_filename"] = saved_filename
            except Exception as persist_err:
                db.rollback()
                file_path.unlink(missing_ok=True)
                print(f"Could not persist check {file_id}: {persist_err}")

        results.append(file_result)

    return {
        "success": True,
        "message": f"Successfully evaluated {len(results)} file(s) against secure ledger.",
        "files": results
    }


# ---------------------------------------------------------------------------
# On-chain anchoring
# ---------------------------------------------------------------------------

def _anchor_asset(asset_id: str, sha256_hex: str) -> dict:
    """Submit an anchor for an asset and record the outcome in its custody log.

    Opens its own session, because this also runs as a background task - by then
    the request's session has been closed.
    """
    result = blockchain.anchor(sha256_hex)
    outcome = result.get("status")
    subject_id = uuid.UUID(asset_id)

    session = SessionLocal()
    try:
        if outcome == "pending":
            entry = custody.append_entry(
                session,
                subject_type=custody.SUBJECT_ASSET,
                subject_id=subject_id,
                event_type=custody.ANCHOR_REQUESTED,
                payload={"sha256": sha256_hex, "tx_hash": result.get("tx_hash")},
                anchor={"tx_hash": result.get("tx_hash"), "status": "pending"},
            )
            session.commit()

            # The receipt may already exist on a fast chain; on a real network it
            # will not, and the retry endpoint picks it up later.
            confirmed = blockchain.confirm(result["tx_hash"])
            if confirmed.get("status") == "confirmed":
                entry.anchor_block = confirmed.get("block_number")
                entry.anchor_status = "confirmed"
                custody.append_entry(
                    session,
                    subject_type=custody.SUBJECT_ASSET,
                    subject_id=subject_id,
                    event_type=custody.ANCHOR_CONFIRMED,
                    payload={
                        "sha256": sha256_hex,
                        "tx_hash": result.get("tx_hash"),
                        "block_number": confirmed.get("block_number"),
                    },
                    anchor={
                        "tx_hash": result.get("tx_hash"),
                        "block_number": confirmed.get("block_number"),
                        "status": "confirmed",
                    },
                )
                session.commit()
                return {
                    "status": "confirmed",
                    "tx_hash": result.get("tx_hash"),
                    "block_number": confirmed.get("block_number"),
                    "explorer": confirmed.get("explorer"),
                }

        elif outcome == "confirmed":
            custody.append_entry(
                session,
                subject_type=custody.SUBJECT_ASSET,
                subject_id=subject_id,
                event_type=custody.ANCHOR_CONFIRMED,
                payload={
                    "sha256": sha256_hex,
                    "already_anchored": True,
                    "timestamp": result.get("timestamp"),
                },
                anchor={"status": "confirmed"},
            )
            session.commit()
            return {"status": "confirmed", "already_anchored": True}

        else:  # 'skipped' when unconfigured, 'failed' when the chain refused
            custody.append_entry(
                session,
                subject_type=custody.SUBJECT_ASSET,
                subject_id=subject_id,
                event_type=custody.ANCHOR_FAILED,
                payload={
                    "sha256": sha256_hex,
                    "outcome": outcome,
                    "reason": result.get("reason"),
                },
                anchor={"status": "failed"},
            )
            session.commit()
            return {"status": outcome, "reason": result.get("reason")}
    except Exception as exc:
        session.rollback()
        print(f"Could not record anchoring for {asset_id}: {exc}")
        return {"status": "error", "reason": str(exc)}
    finally:
        session.close()

    return result


@app.post("/assets/{asset_id}/anchor")
def anchor_asset(asset_id: str, db: Session = Depends(get_db)):
    """Anchor an asset now, or retry an earlier failure."""
    try:
        uuid.UUID(asset_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid asset id.")

    row = db.execute(
        text("SELECT sha256_hash FROM original_assets WHERE id = :id"),
        {"id": asset_id},
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Asset not found.")

    return _anchor_asset(asset_id, row.sha256_hash)


@app.get("/assets")
def list_assets(limit: int = 100, offset: int = 0, db: Session = Depends(get_db)):
    """Registered originals, newest first. Never returns `file_data`.

    The registry view and the "check against" picker both need this. Without it
    the frontend could only show what it had uploaded in the current browser
    session, which is also why picking a specific original was impossible.
    """
    rows = db.execute(
        text("""
            SELECT id, file_name, phash, sha256_hash, created_at
            FROM original_assets
            ORDER BY created_at DESC
            LIMIT :limit OFFSET :offset;
        """),
        {"limit": limit, "offset": offset},
    ).fetchall()

    total = db.execute(text("SELECT count(*) FROM original_assets;")).scalar()

    return {
        "total": total,
        "limit": limit,
        "offset": offset,
        "assets": [
            {
                "id": str(row.id),
                "file_name": row.file_name,
                "phash": row.phash,
                "sha256": row.sha256_hash,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "image_url": f"/view-asset/{row.id}",
            }
            for row in rows
        ],
    }


# ---------------------------------------------------------------------------
# Difference map - "where did it change?"
# ---------------------------------------------------------------------------
# The verdict tells you *that* the evidence differs from the original. These two
# routes tell you *where*, by warping the original into the evidence's frame
# with the homography ORB already recovered, differencing the two, and
# thresholding the result. Results are cached beside the evidence file, keyed by
# (evidence, original), so the second and later requests are free.

def _load_verified_evidence(file_id: str, db: Session) -> tuple[bytes, str]:
    """Return (bytes, saved_filename) for a stored evidence copy, verified.

    The file is located through its database row rather than by searching the
    folder, and its SHA-256 is recomputed and compared with the one recorded
    when the check ran. A copy that was altered, replaced or corrupted on disk
    is refused instead of being shown as if it were the evidence.

    The id arrives from the client, so it is validated as a UUID first - that
    removes any possibility of a crafted value escaping the uploads directory.
    """
    try:
        uuid.UUID(file_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid evidence id.")

    row = db.execute(
        text("SELECT sha256_hash, saved_filename FROM evidence_checks WHERE id = :id"),
        {"id": file_id},
    ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail="Evidence not found.")

    path = UPLOAD_DIR / row.saved_filename
    if not path.is_file():
        raise HTTPException(status_code=404, detail="Evidence file not found.")

    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != row.sha256_hash:
        raise HTTPException(
            status_code=409,
            detail=(
                "The stored evidence file no longer matches the fingerprint "
                "recorded when it was checked, so it will not be shown."
            ),
        )
    return data, row.saved_filename


@app.get("/evidence/{file_id}/image")
def get_evidence_image(file_id: str, db: Session = Depends(get_db)):
    """The evidence copy saved during a check, served only if it is intact."""
    data, saved_filename = _load_verified_evidence(file_id, db)
    media_type = mimetypes.guess_type(saved_filename)[0] or "application/octet-stream"
    return Response(
        content=data,
        media_type=media_type,
        # Never let the browser sniff the bytes into something executable.
        headers={"X-Content-Type-Options": "nosniff"},
    )


def _build_difference_artifacts(file_id: str, original_asset_id: str, db: Session):
    """Return (payload, overlay_path|None) for one evidence/original pair."""
    try:
        uuid.UUID(original_asset_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid original_asset_id.")

    # Verified before the cache is consulted: the overlay is a picture of the
    # evidence, so it must not be shown once the evidence itself fails.
    evidence_bytes, _ = _load_verified_evidence(file_id, db)

    regions_path = UPLOAD_DIR / f"{file_id}__{original_asset_id}__regions.json"
    overlay_path = UPLOAD_DIR / f"{file_id}__{original_asset_id}__overlay.png"

    if regions_path.exists():
        overlay = overlay_path if overlay_path.exists() else None
        return json.loads(regions_path.read_text()), overlay

    original = db.execute(
        text("SELECT file_data FROM original_assets WHERE id = :id"),
        {"id": original_asset_id},
    ).fetchone()
    if original is None or original.file_data is None:
        raise HTTPException(status_code=404, detail="Original asset not found.")

    evidence_bgr = matching.decode_bgr(evidence_bytes)
    original_bgr = matching.decode_bgr(bytes(original.file_data))
    if evidence_bgr is None or original_bgr is None:
        raise HTTPException(status_code=422, detail="Could not decode image.")

    height, width = evidence_bgr.shape[:2]
    result = matching.match_images(original_bgr, evidence_bgr)

    if not result["ok"] or result["homography"] is None:
        payload = {
            "file_id": file_id,
            "original_asset_id": original_asset_id,
            "matched": False,
            "reason": result["reason"],
            "width": width,
            "height": height,
            "regions": [],
        }
        regions_path.write_text(json.dumps(payload, indent=2))
        return payload, None

    _, mask, boxes = matching.difference_map(
        original_bgr, evidence_bgr, result["homography"]
    )
    payload = {
        "file_id": file_id,
        "original_asset_id": original_asset_id,
        "matched": True,
        "width": width,
        "height": height,
        "orb_inliers": result["inliers"],
        "orb_inlier_ratio": result["inlier_ratio"],
        "regions": [
            {
                "x": x, "y": y, "w": w, "h": h,
                "x_pct": round(x / width * 100, 2),
                "y_pct": round(y / height * 100, 2),
                "w_pct": round(w / width * 100, 2),
                "h_pct": round(h / height * 100, 2),
            }
            for x, y, w, h in boxes
        ],
    }
    regions_path.write_text(json.dumps(payload, indent=2))

    overlay_png = matching.render_overlay_png(evidence_bgr, mask)
    if overlay_png is not None:
        overlay_path.write_bytes(overlay_png)

    return payload, (overlay_path if overlay_path.exists() else None)


@app.get("/evidence/{file_id}/altered-regions")
def get_altered_regions(
    file_id: str, original_asset_id: str, db: Session = Depends(get_db)
):
    """Rectangles where the evidence differs from the original it matched.

    Reported in pixels and as percentages of the analysed frame. Matching runs
    on a downscaled copy (long edge capped by matching.MATCH_MAX_EDGE), so the
    percentages are the resolution-independent values to lay out UI with.
    """
    payload, _ = _build_difference_artifacts(file_id, original_asset_id, db)
    return payload


@app.get("/evidence/{file_id}/difference-overlay")
def get_difference_overlay(
    file_id: str, original_asset_id: str, db: Session = Depends(get_db)
):
    """The evidence image with the differing pixels tinted red, as a PNG."""
    _, overlay_path = _build_difference_artifacts(file_id, original_asset_id, db)
    if overlay_path is None:
        raise HTTPException(
            status_code=404,
            detail=(
                "No difference overlay: the evidence did not geometrically "
                "match the original."
            ),
        )
    return FileResponse(overlay_path, media_type="image/png")


# ---------------------------------------------------------------------------
# Chain of custody
# ---------------------------------------------------------------------------

VALID_SUBJECT_TYPES = {custody.SUBJECT_ASSET, custody.SUBJECT_EVIDENCE}


def _validate_subject(subject_type: str, subject_id: str):
    if subject_type not in VALID_SUBJECT_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"subject_type must be one of {sorted(VALID_SUBJECT_TYPES)}.",
        )
    try:
        uuid.UUID(subject_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid subject id.")


def _reconcile_pending_anchors(db, subject_type: str, subject_id: str) -> bool:
    """Confirm an anchor that is still marked pending, if the chain has mined it.

    A Sepolia block takes about 12 seconds, so the receipt is essentially never
    available at the moment registration returns - the entry is written as
    `pending` and the receipt has to come from somewhere later.

    Rather than run a background poller, the receipt is picked up the first time
    anyone looks. That keeps the record self-healing with no extra process, and
    it is why reading a custody log can write: appending the confirmation entry
    is the correct way to record it, since the original entry is immutable.
    """
    pending = db.execute(
        text("""
            SELECT id, anchor_tx_hash
            FROM custody_entries
            WHERE subject_type = :subject_type
              AND subject_id = :subject_id
              AND anchor_status = 'pending'
              AND anchor_tx_hash IS NOT NULL
            ORDER BY seq DESC
            LIMIT 1
        """),
        {"subject_type": subject_type, "subject_id": str(subject_id)},
    ).fetchone()

    if pending is None:
        return False

    result = blockchain.confirm(pending.anchor_tx_hash)
    if result.get("status") != "confirmed":
        return False

    entry = db.get(CustodyEntry, pending.id)
    if entry is not None:
        # The only columns a trigger allows to change after insert.
        entry.anchor_status = "confirmed"
        entry.anchor_block = result.get("block_number")

    custody.append_entry(
        db,
        subject_type=subject_type,
        subject_id=subject_id,
        event_type=custody.ANCHOR_CONFIRMED,
        payload={
            "tx_hash": pending.anchor_tx_hash,
            "block_number": result.get("block_number"),
            "reconciled_on_read": True,
        },
        anchor={
            "tx_hash": pending.anchor_tx_hash,
            "block_number": result.get("block_number"),
            "status": "confirmed",
        },
    )
    db.commit()
    return True


@app.get("/custody/{subject_type}/{subject_id}")
def get_custody(subject_type: str, subject_id: str, db: Session = Depends(get_db)):
    """The full handling history of one asset or one piece of evidence."""
    _validate_subject(subject_type, subject_id)

    # A pending anchor may have been mined since it was written. Reconcile
    # before reading so the returned record is current; a reconciliation
    # failure must not stop the log from being returned.
    try:
        _reconcile_pending_anchors(db, subject_type, subject_id)
    except Exception as exc:
        db.rollback()
        print(f"Anchor reconciliation failed for {subject_id}: {exc}")

    entries = custody.list_entries(db, subject_type, subject_id)
    return {
        "subject_type": subject_type,
        "subject_id": subject_id,
        "count": len(entries),
        "entries": entries,
    }


@app.get("/custody/{subject_type}/{subject_id}/verify")
def verify_custody(subject_type: str, subject_id: str, db: Session = Depends(get_db)):
    """Recompute the chain and report whether it is intact.

    This is what to run if the record is suspected of having been edited. Every
    entry hash is recomputed from that entry's contents plus its predecessor's
    hash, so altering one entry breaks its own hash and every hash after it.
    """
    _validate_subject(subject_type, subject_id)
    entries = custody.list_entries(db, subject_type, subject_id)
    report = custody.verify_chain(entries)
    report.update({"subject_type": subject_type, "subject_id": subject_id})
    return report


@app.get("/checks")
def list_checks(limit: int = 50, offset: int = 0, db: Session = Depends(get_db)):
    """Recent comparisons, newest first."""
    rows = db.execute(
        text("""
            SELECT c.id, c.verdict, c.match_method, c.orb_inliers, c.saved_filename,
                   c.created_at, a.file_name AS asset_name
            FROM evidence_checks c
            LEFT JOIN original_assets a ON a.id = c.asset_id
            ORDER BY c.created_at DESC
            LIMIT :limit OFFSET :offset
        """),
        {"limit": limit, "offset": offset},
    ).fetchall()

    total = db.execute(text("SELECT count(*) FROM evidence_checks;")).scalar()

    return {
        "total": total,
        "checks": [
            {
                "file_id": str(row.id),
                "verification_status": row.verdict,
                "match_method": row.match_method,
                "orb_inliers": row.orb_inliers,
                "saved_filename": row.saved_filename,
                "matched_asset_name": row.asset_name,
                "created_at": row.created_at.isoformat() if row.created_at else None,
            }
            for row in rows
        ],
    }


@app.get("/me")
def whoami(user: dict | None = Depends(auth.optional_user)):
    """Who the API believes you are.

    Exists so the frontend can prove a token round-trips, and so a failure to
    authenticate is visible as a 401 rather than as a silent downgrade to
    anonymous.
    """
    if user is None:
        return {"authenticated": False, "auth_configured": auth.is_configured()}

    return {
        "authenticated": True,
        "actor": auth.actor_from(user),
        "id": user.get("sub") or user.get("id"),
        "email": user.get("email"),
        "name": user.get("name"),
        "email_verified": user.get("emailVerified"),
        "role": user.get("role"),
        "expires_at": user.get("exp"),
    }


@app.get("/verify-anchor/{sha256}")
def verify_anchor(sha256: str, db: Session = Depends(get_db)):
    """Independently check a digest: what the database claims, what the chain says.

    This is the endpoint a third party can use without trusting the rest of this
    application. It compares three things - the registry, the on-chain anchor,
    and the integrity of the local custody chain - and reports where they agree
    and where they do not.

    Disagreement is the interesting case. `anchored_not_registered` means a
    digest is on-chain that this database has no record of, which is exactly the
    signal you would want if someone were trying to hide a registration.
    """
    digest = (sha256 or "").strip().lower()
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise HTTPException(
            status_code=400,
            detail="Expected a 64-character hexadecimal SHA-256 digest.",
        )

    row = db.execute(
        text("""
            SELECT id, file_name, phash, created_at
            FROM original_assets
            WHERE sha256_hash = :digest
            ORDER BY created_at ASC
            LIMIT 1
        """),
        {"digest": digest},
    ).fetchone()

    registered = row is not None
    chain = blockchain.get_anchor(digest)

    custody_report = None
    if registered:
        entries = custody.list_entries(db, custody.SUBJECT_ASSET, row.id)
        custody_report = custody.verify_chain(entries)

    anchored = bool(chain.get("anchored"))
    if registered and anchored:
        state = "registered_and_anchored"
    elif registered:
        state = "registered_not_anchored"
    elif anchored:
        state = "anchored_not_registered"
    else:
        state = "not_found"

    explanations = {
        "registered_and_anchored": (
            "This file is in the registry and its fingerprint is anchored on-chain. "
            "The record can be checked by anyone without trusting ForenScreen."
        ),
        "registered_not_anchored": (
            "This file is in the registry, but its fingerprint is not on-chain. "
            "The record is only as trustworthy as this database."
        ),
        "anchored_not_registered": (
            "This fingerprint is anchored on-chain but missing from the registry. "
            "That mismatch is worth investigating."
        ),
        "not_found": "This fingerprint is neither registered nor anchored.",
    }

    return {
        "sha256": digest,
        "state": state,
        "explanation": explanations[state],
        "database": (
            {
                "registered": True,
                "asset_id": str(row.id),
                "file_name": row.file_name,
                "phash": row.phash,
                "registered_at": row.created_at.isoformat() if row.created_at else None,
            }
            if registered
            else {"registered": False}
        ),
        "chain": chain,
        "custody_chain": custody_report,
    }


@app.post("/checks/{check_id}/report")
def export_check_report(
    check_id: str,
    user: dict | None = Depends(auth.optional_user),
    db: Session = Depends(get_db),
):
    """Produce a report for one check, and record that it was exported.

    Exporting is itself a custody event: a copy leaving the system is exactly
    the kind of thing a chain of custody exists to record. The report embeds the
    custody chain and its verification result, so the document carries its own
    evidence of completeness instead of asserting it.
    """
    try:
        uuid.UUID(check_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid check id.")

    row = db.execute(
        text("""
            SELECT c.id, c.asset_id, c.sha256_hash, c.phash, c.verdict,
                   c.match_method, c.orb_inliers, c.orb_inlier_ratio,
                   c.phash_distance, c.saved_filename, c.created_at,
                   a.file_name AS asset_name, a.sha256_hash AS asset_sha256
            FROM evidence_checks c
            LEFT JOIN original_assets a ON a.id = c.asset_id
            WHERE c.id = :id
        """),
        {"id": check_id},
    ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Check not found.")

    entries = custody.list_entries(db, custody.SUBJECT_EVIDENCE, check_id)
    chain = custody.verify_chain(entries)

    custody.append_entry(
        db,
        subject_type=custody.SUBJECT_EVIDENCE,
        subject_id=uuid.UUID(check_id),
        event_type=custody.REPORT_EXPORTED,
        actor=auth.actor_from(user),
        payload={
            "verdict": row.verdict,
            "custody_entries": len(entries),
            "chain_head": chain.get("head_hash"),
        },
    )
    db.commit()

    return {
        "report": {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "generated_by": auth.actor_from(user),
            "check": {
                "id": str(row.id),
                "sha256": row.sha256_hash,
                "phash": row.phash,
                "verdict": row.verdict,
                "match_method": row.match_method,
                "orb_inliers": row.orb_inliers,
                "orb_inlier_ratio": (
                    float(row.orb_inlier_ratio) if row.orb_inlier_ratio is not None else None
                ),
                "phash_distance": row.phash_distance,
                "saved_filename": row.saved_filename,
                "checked_at": row.created_at.isoformat() if row.created_at else None,
                "matched_asset": (
                    {
                        "id": str(row.asset_id),
                        "file_name": row.asset_name,
                        "sha256": row.asset_sha256,
                    }
                    if row.asset_id
                    else None
                ),
            },
            "custody": {"chain": chain, "entries": entries},
        }
    }


@app.get("/checks/{check_id}")
def get_check(check_id: str, db: Session = Depends(get_db)):
    """A stored comparison, so a result survives a page reload."""
    try:
        uuid.UUID(check_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid check id.")

    row = db.execute(
        text("""
            SELECT id, asset_id, sha256_hash, phash, verdict, match_method,
                   orb_inliers, orb_inlier_ratio, phash_distance,
                   saved_filename, created_at
            FROM evidence_checks
            WHERE id = :id
        """),
        {"id": check_id},
    ).fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail="Check not found.")

    return {
        "file_id": str(row.id),
        "asset_id": str(row.asset_id) if row.asset_id else None,
        "sha256_hash": row.sha256_hash,
        "phash": row.phash,
        "verification_status": row.verdict,
        "match_method": row.match_method,
        "orb_inliers": row.orb_inliers,
        "orb_inlier_ratio": (
            float(row.orb_inlier_ratio) if row.orb_inlier_ratio is not None else None
        ),
        "phash_distance": row.phash_distance,
        "saved_filename": row.saved_filename,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
