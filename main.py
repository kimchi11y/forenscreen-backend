from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import hashlib
from datetime import datetime
import uuid
from typing import List, Dict
from PIL import Image, ImageChops, ImageStat
from PIL import UnidentifiedImageError
from PIL.ExifTags import TAGS, GPSTAGS
import io
import imagehash
import numpy as np
from pillow_heif import register_heif_opener
from collections import Counter
import os
from pathlib import Path
import json
from sqlalchemy import text

app = FastAPI()

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# Persistent storage for duplicate detection
REGISTRY_FILE = UPLOAD_DIR / "file_registry.json"

def load_file_registry():
    """Load file registry from disk for persistent duplicate detection."""
    if REGISTRY_FILE.exists():
        with open(REGISTRY_FILE, 'r') as f:
            return json.load(f)
    return {}

def save_file_registry(registry):
    """Save file registry to disk."""
    with open(REGISTRY_FILE, 'w') as f:
        json.dump(registry, f, indent=2)

# Initialize registry from disk
uploaded_files_registry = load_file_registry()

# Mount static files for serving uploaded images
app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:5174", "http://localhost:3000"],
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


# Global dictionary to track uploaded files for duplicate detection
uploaded_files_registry: Dict[str, Dict] = {}


def analyze_screenshot_forensics(image_content: bytes, filename: str) -> dict:
    """
    Perform forensic analysis on screenshot images.
    Includes edit detection, file analysis, and duplicate detection.
    """
    forensic_data = {
        "edit_detection": {},
        "image_analysis": {},
        "duplicate_check": {},
        "timeline_data": {},
        "manipulation_indicators": []
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

        # Color analysis
        forensic_data["image_analysis"]["color_profile"] = analyze_color_distribution(image)

        # Error Level Analysis (ELA) - detects edited regions
        forensic_data["edit_detection"] = perform_error_level_analysis(image)

        # Compression analysis
        forensic_data["edit_detection"]["compression_analysis"] = analyze_compression_artifacts(image)

        # Duplicate detection using perceptual hashing
        forensic_data["duplicate_check"] = check_for_duplicates(image, filename)

        # File timeline data
        forensic_data["timeline_data"] = {
            "analysis_timestamp": datetime.now().isoformat(),
            "file_size_bytes": len(image_content),
            "file_size_mb": round(len(image_content) / (1024 * 1024), 2),
            "format": image.format
        }

        # Manipulation indicators
        forensic_data["manipulation_indicators"] = detect_manipulation_indicators(
            forensic_data["edit_detection"],
            forensic_data["image_analysis"]
        )

    except Exception as e:
        forensic_data["analysis_error"] = str(e)

    return forensic_data


def analyze_color_distribution(image: Image.Image) -> dict:
    """Analyze color distribution in the image."""
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Sample colors (resize for performance)
        small_img = image.resize((100, 100))
        pixels = list(small_img.getdata())

        # Count color frequencies
        color_counter = Counter(pixels)
        dominant_colors = color_counter.most_common(5)

        # Calculate statistics
        r_values = [p[0] for p in pixels]
        g_values = [p[1] for p in pixels]
        b_values = [p[2] for p in pixels]

        return {
            "dominant_colors": [f"rgb({c[0]},{c[1]},{c[2]})" for c in dominant_colors],
            "brightness_avg": round(sum(r_values + g_values + b_values) / (len(pixels) * 3), 2),
            "color_variance": round(np.var([sum(p)/3 for p in pixels]), 2)
        }
    except Exception as e:
        return {"error": str(e)}


# ELA thresholds are expressed as a PERCENTAGE of the maximum possible pixel
# difference (0-255), not as a raw 0-100 score. The raw `stat.mean` diff used
# below ranges 0-255, so it must be normalized before being compared against
# a 0-100 scale or displayed as "X/100" - treating the raw value as already
# being 0-100 was the root cause of ELA scores/likelihoods looking wrong
# (nearly everything read as "High").
#
# These thresholds are still heuristic defaults and should be recalibrated
# against a labeled set of known-authentic vs. known-tampered images before
# being relied on for a real verdict (see FYP objective: "evaluate the
# accuracy of manipulation detection").
ELA_THRESHOLD_HIGH_PCT = 20.0   # % of max possible pixel error
ELA_THRESHOLD_MEDIUM_PCT = 10.0

def perform_error_level_analysis(image: Image.Image, quality: int = 90) -> dict:
    """
    Perform Error Level Analysis to detect edited regions.
    Edited areas often have different compression artifacts.

    Note: ELA is designed to catch *local* recompression mismatches within an
    image that already has a JPEG compression history (i.e. edited regions
    were saved at a different quality than the rest of the image). For
    lossless sources (e.g. PNG screenshots) there is no original JPEG quality
    to diff against, so a single fixed recompression pass will produce
    elevated, roughly uniform error across the whole image (especially at
    sharp text/UI edges) regardless of tampering. Treat this score as a
    supporting signal, not a standalone verdict.
    """
    try:
        # Convert to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Save with specific quality and reload
        buffer = io.BytesIO()
        image.save(buffer, format='JPEG', quality=quality)
        buffer.seek(0)
        compressed = Image.open(buffer)

        # Calculate difference
        diff = ImageChops.difference(image, compressed)

        # Get statistics. Raw mean pixel difference is on a 0-255 scale.
        stat = ImageStat.Stat(diff)
        ela_score_raw = round(sum(stat.mean) / len(stat.mean), 2)
        # NOTE: Pillow's ImageStat.Stat exposes variance as `.var`, not
        # `.variance`. The old `.variance` attribute reference raised an
        # AttributeError on every call, which was silently swallowed by the
        # except block below - meaning ELA was never actually computing
        # successfully in production and always fell back to
        # {"edit_likelihood": "Unknown"}. This was the primary cause of the
        # ELA numbers looking wrong.
        ela_variance = round(sum(stat.var) / len(stat.var), 2)

        # Normalize to a proper 0-100 scale for display/comparison.
        ela_score_normalized = round((ela_score_raw / 255) * 100, 2)

        # Determine if editing likely occurred, using the normalized score
        # against percentage-based thresholds.
        edit_likelihood = "Low"
        if ela_score_normalized > ELA_THRESHOLD_HIGH_PCT:
            edit_likelihood = "High"
        elif ela_score_normalized > ELA_THRESHOLD_MEDIUM_PCT:
            edit_likelihood = "Medium"

        return {
            # Raw 0-255 mean pixel error (kept for debugging/back-compat).
            "ela_score": ela_score_raw,
            # Correctly-scaled 0-100 score - use this for display/thresholds.
            "ela_score_normalized": ela_score_normalized,
            "ela_variance": ela_variance,
            "edit_likelihood": edit_likelihood,
            "interpretation": (
                f"Score {ela_score_normalized}/100 (normalized) - "
                f"{edit_likelihood} likelihood of manipulation. "
                f"Supporting signal only - not the sole basis for a tamper verdict."
            )
        }
    except Exception as e:
        return {"error": str(e), "edit_likelihood": "Unknown"}


def analyze_compression_artifacts(image: Image.Image) -> dict:
    """Analyze compression artifacts and inconsistencies."""
    try:
        # Check format-specific compression
        if image.format == 'JPEG':
            # JPEG has multiple quality levels detectable
            return {
                "compression_type": "JPEG",
                "has_multiple_compression": False,  # Simplified
                "artifact_level": "Standard"
            }
        else:
            return {
                "compression_type": image.format or "Unknown",
                "artifact_level": "Low"
            }
    except Exception as e:
        return {"error": str(e)}
import hashlib
import io
from fastapi import BackgroundTasks, FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse, StreamingResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from PIL import Image
import imagehash

import blockchain
import custody
import matching
from database import SessionLocal, ensure_schema, get_db
from models import EvidenceCheck, OriginalAsset

# Pillow does not decode HEIC/HEIF files unless this plugin is registered.
register_heif_opener()

# Ensure tables exist. Tolerates an unreachable database so the process still
# starts and can serve /health instead of hanging on connect.
ensure_schema()

@app.post("/upload-asset/", status_code=status.HTTP_201_CREATED)
async def upload_original_asset(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
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
            payload={
                "sha256": sha256_sig,
                "phash": calculated_phash,
                "file_name": file.filename,
                "size_bytes": len(file_bytes),
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

def check_for_duplicates(image: Image.Image, filename: str) -> dict:
    """
    Check for duplicate or similar images using perceptual hashing.
    Detects if same screenshot was uploaded multiple times.
    """
    try:
        # Calculate perceptual hashes
        ahash = str(imagehash.average_hash(image, hash_size=8))
        phash = str(imagehash.phash(image, hash_size=8))
        dhash = str(imagehash.dhash(image, hash_size=8))
        whash = str(imagehash.whash(image, hash_size=8))

        # Check against registry
        duplicate_info = {
            "is_duplicate": False,
            "is_similar": False,
            "similar_files": [],
            "hashes": {
                "average_hash": ahash,
                "perceptual_hash": phash,
                "difference_hash": dhash,
                "wavelet_hash": whash
            }
        }

        # Check for exact matches (all hashes same)
        for reg_filename, reg_data in uploaded_files_registry.items():
            reg_hashes = reg_data.get("hashes", {})
            if (reg_hashes.get("perceptual_hash") == phash and
                reg_hashes.get("average_hash") == ahash):
                duplicate_info["is_duplicate"] = True
                duplicate_info["similar_files"].append({
                    "filename": reg_filename,
                    "similarity": "100% (Exact Match)",
                    "upload_time": reg_data.get("upload_time")
                })

        # Check for similar images (partial hash match)
        if not duplicate_info["is_duplicate"]:
            for reg_filename, reg_data in uploaded_files_registry.items():
                # Simple hamming distance simulation
                reg_phash = reg_data.get("hashes", {}).get("perceptual_hash", "")
                if reg_phash and phash:
                    # Count differing characters
                    diff = sum(c1 != c2 for c1, c2 in zip(reg_phash, phash))
                    similarity = round((64 - diff) / 64 * 100, 1)  # Assuming 64-char hash

                    if similarity >= 95:  # 95%+ similarity
                        duplicate_info["is_similar"] = True
                        duplicate_info["similar_files"].append({
                            "filename": reg_filename,
                            "similarity": f"{similarity}%",
                            "upload_time": reg_data.get("upload_time")
                        })

        # Register this image
        uploaded_files_registry[filename] = {
            "hashes": duplicate_info["hashes"],
            "upload_time": datetime.now().isoformat()
        }

        # Save registry to disk for persistence
        save_file_registry(uploaded_files_registry)

        return duplicate_info
    except Exception as e:
        return {"error": str(e), "is_duplicate": False}


def detect_manipulation_indicators(edit_data: dict, image_data: dict) -> list:
    """Generate human-readable manipulation indicators."""
    indicators = []

    # ELA-based indicators (use the normalized 0-100 score, not the raw
    # 0-255 value - see perform_error_level_analysis for why this matters).
    ela_score_normalized = edit_data.get("ela_score_normalized", 0)
    if ela_score_normalized > ELA_THRESHOLD_HIGH_PCT:
        indicators.append("⚠️ High ELA score suggests possible image editing")
    elif ela_score_normalized > ELA_THRESHOLD_MEDIUM_PCT:
        indicators.append("⚡ Medium ELA score - minor modifications possible")

    # Color-based indicators
    if image_data.get("color_profile", {}).get("color_variance", 0) < 100:
        indicators.append("🎨 Low color variance - may indicate screenshot or digital content")

    # Format-based indicators
    format_info = image_data.get("format", "")
    if format_info == "PNG":
        indicators.append("📄 PNG format - typical for screenshots, preserves detail")

    # Compression indicators
    comp_analysis = edit_data.get("compression_analysis", {})
    if comp_analysis.get("compression_type") == "JPEG":
        indicators.append("🗜️ JPEG compression - quality loss may obscure edits")

    if not indicators:
        indicators.append("✅ No obvious manipulation indicators detected")

    return indicators

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
            continue

        # Calculate SHA-256 hash for strict file integrity matching
        sha256_hash = hashlib.sha256(file_content).hexdigest()
        file_id = str(uuid.uuid4())

        # Keep your local disk save step for now (as a temporary backup)
        file_extension = Path(file.filename).suffix
        saved_filename = f"{file_id}{file_extension}"
        file_path = UPLOAD_DIR / saved_filename
        with open(file_path, "wb") as f:
            f.write(file_content)

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

        # Perform visual similarity search if the uploaded file is an image
        if file.content_type and file.content_type.startswith('image/'):
            try:
                # 2. Compute the Visual Perceptual Hash (pHash) in memory
                image = Image.open(io.BytesIO(file_content))
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

                # Run your existing screenshot forensics pipeline
                # (Passing file_content or updating your method internally)
                forensic_data = analyze_screenshot_forensics(file_content, file.filename)
                
            except Exception as img_err:
                print(f"Error processing visual forensics for {file.filename}: {img_err}")
                verification_status = "PROCESSING_ERROR"

        # Create comprehensive result payload for each file
        file_result = {
            "file_id": file_id,
            "filename": file.filename,
            "saved_filename": saved_filename,
            "file_path": str(file_path),
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
        file_result["persisted"] = False
        if computed_phash is not None:
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
                    payload={
                        "sha256": sha256_hash,
                        "verdict": verification_status,
                        "match_method": (matched_asset_info or {}).get("match_method"),
                        "orb_inliers": (matched_asset_info or {}).get("orb_inliers"),
                        "matched_asset_id": matched_asset_id,
                        "saved_filename": saved_filename,
                    },
                )
                db.commit()
                file_result["persisted"] = True
            except Exception as persist_err:
                db.rollback()
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

def _evidence_path(file_id: str):
    """Resolve an evidence id to its stored file.

    The id arrives from the client, so it is validated as a UUID first - that
    removes any possibility of a crafted value escaping the uploads directory.
    """
    try:
        uuid.UUID(file_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid evidence id.")

    for candidate in UPLOAD_DIR.glob(f"{file_id}.*"):
        if "__" not in candidate.name:  # skip our own cache sidecars
            return candidate
    return None


def _build_difference_artifacts(file_id: str, original_asset_id: str, db: Session):
    """Return (payload, overlay_path|None) for one evidence/original pair."""
    try:
        uuid.UUID(original_asset_id)
    except (ValueError, AttributeError, TypeError):
        raise HTTPException(status_code=400, detail="Invalid original_asset_id.")

    regions_path = UPLOAD_DIR / f"{file_id}__{original_asset_id}__regions.json"
    overlay_path = UPLOAD_DIR / f"{file_id}__{original_asset_id}__overlay.png"

    if regions_path.exists():
        overlay = overlay_path if overlay_path.exists() else None
        return json.loads(regions_path.read_text()), overlay

    evidence_file = _evidence_path(file_id)
    if evidence_file is None:
        raise HTTPException(status_code=404, detail="Evidence file not found.")

    original = db.execute(
        text("SELECT file_data FROM original_assets WHERE id = :id"),
        {"id": original_asset_id},
    ).fetchone()
    if original is None or original.file_data is None:
        raise HTTPException(status_code=404, detail="Original asset not found.")

    evidence_bgr = matching.decode_bgr(evidence_file.read_bytes())
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


@app.get("/custody/{subject_type}/{subject_id}")
def get_custody(subject_type: str, subject_id: str, db: Session = Depends(get_db)):
    """The full handling history of one asset or one piece of evidence."""
    _validate_subject(subject_type, subject_id)
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
