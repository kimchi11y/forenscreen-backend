from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import hashlib
from datetime import datetime
import uuid
from typing import List, Dict
from PIL import Image, ImageChops, ImageStat
from PIL.ExifTags import TAGS, GPSTAGS
import io
import imagehash
import numpy as np
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
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from PIL import Image
import imagehash

from database import get_db, engine, Base
from models import OriginalAsset

# Dynamically ensure tables are built (if they don't already exist)
Base.metadata.create_all(bind=engine)

@app.post("/upload-asset/", status_code=status.HTTP_201_CREATED)
async def upload_original_asset(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # 1. Read the raw incoming file data into memory
    file_bytes = await file.read()
    
    if not file_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        # 2. Compute Strict Cryptographic Signature (SHA-256)
        sha256_sig = hashlib.sha256(file_bytes).hexdigest()

        # 3. Compute Visual Perceptual Hash (pHash)
        image = Image.open(io.BytesIO(file_bytes))
        calculated_phash = str(imagehash.phash(image)) # Produces a 16-character hex string

        # 4. Bind information to our database record
        new_asset = OriginalAsset(
            file_name=file.filename,
            file_data=file_bytes, # Storing binary payload directly
            phash=calculated_phash,
            sha256_hash=sha256_sig
        )

        db.add(new_asset)
        db.commit()
        db.refresh(new_asset)

        return {
            "message": "Asset safely stored in Neon DB",
            "id": str(new_asset.id),
            "file_name": new_asset.file_name,
            "phash": new_asset.phash,
            "sha256": new_asset.sha256_hash
        }

    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to process image asset: {str(e)}")


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

# How close two pHash values (in bits, out of 64) must be to be considered
# the same underlying image.
PHASH_MATCH_THRESHOLD = 10
# If the #2 candidate is within this many bits of the #1 candidate, the match
# is ambiguous (could plausibly be either original) and must not be silently
# resolved by picking the numerically-closest one. This is what caused two
# registered originals (A and C) to be confused with each other when
# verifying a tampered derivative of A.
PHASH_AMBIGUITY_MARGIN = 4


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

        # Initialize core verification status variables
        verification_status = "UNKNOWN_ASSET"
        matched_asset_info = None
        forensic_data = {}

        # Perform visual similarity search if the uploaded file is an image
        if file.content_type and file.content_type.startswith('image/'):
            try:
                # 2. Compute the Visual Perceptual Hash (pHash) in memory
                image = Image.open(io.BytesIO(file_content))
                computed_phash = str(imagehash.phash(image))

                match = None
                ambiguous_candidates = None

                if original_asset_id:
                    # Explicit scoping requested: verify against this exact
                    # registered original only, no similarity search needed.
                    scoped_query = text("""
                        SELECT id, file_name, phash, sha256_hash
                        FROM original_assets
                        WHERE id = :asset_id;
                    """)
                    match = db.execute(scoped_query, {"asset_id": original_asset_id}).fetchone()
                else:
                    # 3. NEON DB QUERY: Search original assets for close visual matches
                    # using Postgres bitwise XOR (#). We fetch the top 2 candidates
                    # (not just 1) so we can detect ambiguous matches instead of
                    # blindly trusting whichever one happens to be numerically
                    # closest - two unrelated registered originals (e.g. two
                    # similar-looking screenshots) can easily land within a few
                    # bits of each other in pHash space.
                    search_query = text("""
                        SELECT id, file_name, phash, sha256_hash,
                               BIT_COUNT(('x' || phash)::bit(64) # ('x' || :query_phash)::bit(64)) AS distance
                        FROM original_assets
                        WHERE BIT_COUNT(('x' || phash)::bit(64) # ('x' || :query_phash)::bit(64)) <= :threshold
                        ORDER BY distance ASC
                        LIMIT 2;
                    """)

                    candidates = db.execute(
                        search_query,
                        {"query_phash": computed_phash, "threshold": PHASH_MATCH_THRESHOLD}
                    ).fetchall()

                    if len(candidates) >= 2:
                        best, runner_up = candidates[0], candidates[1]
                        if (runner_up.distance - best.distance) < PHASH_AMBIGUITY_MARGIN:
                            ambiguous_candidates = [
                                {
                                    "original_asset_id": str(c.id),
                                    "original_filename": c.file_name,
                                    "original_sha256": c.sha256_hash,
                                    "phash_distance": c.distance,
                                }
                                for c in candidates
                            ]
                        else:
                            match = best
                    elif len(candidates) == 1:
                        match = candidates[0]

                if ambiguous_candidates:
                    verification_status = "AMBIGUOUS_MATCH"
                    matched_asset_info = {
                        "candidates": ambiguous_candidates,
                        "message": (
                            "Multiple registered originals are visually similar enough "
                            "that this evidence cannot be reliably matched automatically. "
                            "Re-verify against a specific original_asset_id."
                        )
                    }
                elif match:
                    # Map out matching asset metadata retrieved from Neon
                    matched_asset_info = {
                        "original_asset_id": str(match.id),
                        "original_filename": match.file_name,
                        "original_sha256": match.sha256_hash
                    }

                    # 4. CROSS-REFERENCE INTEGRITY: Check if the hashes match bit-by-bit
                    if sha256_hash == match.sha256_hash:
                        verification_status = "VERIFIED_AUTHENTIC"
                    else:
                        verification_status = "TAMPERED_MANIPULATED"
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

        results.append(file_result)

    return {
        "success": True,
        "message": f"Successfully evaluated {len(results)} file(s) against secure ledger.",
        "files": results
    }
