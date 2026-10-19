"""Feature-based evidence matching (ORB keypoints + RANSAC homography).

Why this exists
---------------
A 64-bit pHash answers "does this look broadly like the same picture?" That is
good enough to survive recompression and resizing, but it breaks the moment the
framing changes - a 5% crop or an added platform status bar moves it 12-14 bits,
well past any sane threshold - and it is too coarse to distinguish two different
screenshots of the same app, which can land 0-2 bits apart.

ORB keypoint matching with a RANSAC homography does not have those failure
modes: it is robust to crop, rotation, scale and re-encoding, it rejects
unrelated images because their matches are not geometrically consistent, and it
recovers the transform between the two images - which is what lets us render a
difference map showing *where* the evidence departs from the original.

Division of labour
------------------
    pHash  - cheap pre-filter. Narrows the registry to a handful of candidates.
    ORB    - decides the verdict, and supplies the homography for the diff map.

All functions here are pure CPU work and take/return plain numpy arrays or
bytes, so the caller can push them onto a worker thread without them touching
the database or the event loop.
"""

import io

import cv2
import numpy as np
from PIL import Image
from pillow_heif import register_heif_opener

# Pillow cannot decode HEIC/HEIF without this plugin; the call is idempotent.
register_heif_opener()

# --- keypoint matching ------------------------------------------------------
ORB_MAX_FEATURES = 3000
ORB_FAST_THRESHOLD = 10
LOWE_RATIO = 0.75              # standard ratio test for ORB's binary descriptors
RANSAC_REPROJECTION_PX = 5.0
MIN_GOOD_MATCHES = 12          # below this a homography is not worth estimating
MIN_INLIERS = 30               # absolute floor for "these are the same picture"
MIN_INLIER_RATIO = 0.5         # and the inliers must dominate the raw matches

# Long edge used for matching. Full-resolution screenshots (~1170x2532) cost
# roughly 3x the ORB work for no measurable gain in inlier count, and the
# difference map is only ever displayed scaled down.
MATCH_MAX_EDGE = 1000

# --- difference map ---------------------------------------------------------
DIFF_BLUR = 21
DIFF_THRESHOLD = 30            # 0-255 grey level; absorbs JPEG ringing
DIFF_MIN_AREA_PCT = 0.2        # ignore specks smaller than this % of the frame
OVERLAY_ALPHA = 0.45


def decode_bgr(image_bytes, max_edge=MATCH_MAX_EDGE):
    """Decode any supported image (PNG/JPEG/HEIC) to a BGR array, downscaled.

    Returns None if the bytes are not a decodable image, so callers can treat
    that as "no match" rather than raising.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as im:
            im = im.convert("RGB")
            if max_edge and max(im.size) > max_edge:
                im.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
                im.load()
            arr = np.asarray(im)
    except Exception:
        return None
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _orb():
    return cv2.ORB_create(
        nfeatures=ORB_MAX_FEATURES, fastThreshold=ORB_FAST_THRESHOLD
    )


def _blank_result(ok=False, reason=None, **extra):
    out = {
        "ok": ok,
        "reason": reason,
        "good_matches": 0,
        "inliers": 0,
        "inlier_ratio": 0.0,
        "keypoints_original": 0,
        "keypoints_evidence": 0,
        "homography": None,
    }
    out.update(extra)
    return out


def match_images(original_bgr, evidence_bgr, compute_homography=True):
    """Geometrically match two BGR images.

    The returned ``homography`` maps points in *original* to *evidence*, i.e.
    it can be used with ``cv2.warpPerspective(original, H, evidence_size)`` to
    bring the original into the evidence's frame for differencing.

    A result is ``ok`` when the two images share enough geometrically
    consistent keypoints to be considered the same picture. Note this says
    nothing about the file bytes - that is what SHA-256 is for.
    """
    orb = _orb()
    kp_o, des_o = orb.detectAndCompute(
        cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY), None
    )
    kp_e, des_e = orb.detectAndCompute(
        cv2.cvtColor(evidence_bgr, cv2.COLOR_BGR2GRAY), None
    )

    n_o = 0 if kp_o is None else len(kp_o)
    n_e = 0 if kp_e is None else len(kp_e)

    if des_o is None or des_e is None or n_o < MIN_GOOD_MATCHES or n_e < MIN_GOOD_MATCHES:
        return _blank_result(
            reason="too_few_keypoints",
            keypoints_original=n_o,
            keypoints_evidence=n_e,
        )

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    knn = matcher.knnMatch(des_o, des_e, k=2)

    # Lowe's ratio test: keep a match only if it is clearly better than the
    # runner-up, which discards the ambiguous descriptors that dominate in
    # flat areas of screenshots.
    good = [
        m
        for pair in knn
        if len(pair) == 2
        for m, n in (pair,)
        if m.distance < LOWE_RATIO * n.distance
    ]

    if len(good) < MIN_GOOD_MATCHES:
        return _blank_result(
            reason="too_few_good_matches",
            good_matches=len(good),
            keypoints_original=n_o,
            keypoints_evidence=n_e,
        )

    src = np.float32([kp_o[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_e[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    if not compute_homography:
        return {
            "ok": False,
            "reason": "homography_not_requested",
            "good_matches": len(good),
            "inliers": 0,
            "inlier_ratio": 0.0,
            "keypoints_original": n_o,
            "keypoints_evidence": n_e,
            "homography": None,
        }

    homography, mask = cv2.findHomography(
        src, dst, cv2.RANSAC, RANSAC_REPROJECTION_PX
    )
    if homography is None or mask is None:
        return _blank_result(
            reason="homography_failed",
            good_matches=len(good),
            keypoints_original=n_o,
            keypoints_evidence=n_e,
        )

    inliers = int(mask.sum())
    ratio = inliers / len(good)

    return {
        "ok": inliers >= MIN_INLIERS and ratio >= MIN_INLIER_RATIO,
        "reason": None if inliers >= MIN_INLIERS and ratio >= MIN_INLIER_RATIO
        else "insufficient_consensus",
        "good_matches": len(good),
        "inliers": inliers,
        "inlier_ratio": round(ratio, 3),
        "keypoints_original": n_o,
        "keypoints_evidence": n_e,
        "homography": homography,
    }


def rank_candidates(evidence_bytes, candidates, max_edge=MATCH_MAX_EDGE):
    """Score every candidate original against the evidence, best first.

    ``candidates`` is a list of dicts each carrying at least ``id``,
    ``file_name``, ``sha256_hash``, ``file_data`` and optionally
    ``phash_distance``. Each candidate's metadata is echoed back alongside its
    match result so the caller can build the verdict without another lookup.
    """
    evidence_bgr = decode_bgr(evidence_bytes, max_edge)
    if evidence_bgr is None:
        return []

    ranked = []
    for candidate in candidates:
        original_bgr = decode_bgr(candidate["file_data"], max_edge)
        if original_bgr is None:
            result = _blank_result(reason="undecodable_original")
        else:
            result = match_images(original_bgr, evidence_bgr)

        result["homography"] = None  # never leave numpy state in the payload
        result.update(
            {
                "original_asset_id": candidate["id"],
                "original_filename": candidate["file_name"],
                "original_sha256": candidate["sha256_hash"],
                "phash_distance": candidate.get("phash_distance"),
            }
        )
        ranked.append(result)

    ranked.sort(key=lambda r: (-r["inliers"], -r["inlier_ratio"]))
    return ranked


def difference_map(original_bgr, evidence_bgr, homography):
    """Warp the original into the evidence's frame and locate what differs.

    Returns ``(aligned, mask, boxes)`` where ``boxes`` is a list of
    ``(x, y, w, h)`` in evidence coordinates, largest first.
    """
    height, width = evidence_bgr.shape[:2]
    aligned = cv2.warpPerspective(original_bgr, homography, (width, height))

    delta = cv2.cvtColor(cv2.absdiff(aligned, evidence_bgr), cv2.COLOR_BGR2GRAY)
    delta = cv2.GaussianBlur(delta, (DIFF_BLUR, DIFF_BLUR), 0)

    _, mask = cv2.threshold(delta, DIFF_THRESHOLD, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = width * height * (DIFF_MIN_AREA_PCT / 100.0)
    boxes = [
        cv2.boundingRect(c)
        for c in contours
        if cv2.contourArea(c) >= min_area
    ]
    boxes.sort(key=lambda b: b[2] * b[3], reverse=True)
    return aligned, mask, boxes


def render_overlay_png(evidence_bgr, mask, alpha=OVERLAY_ALPHA):
    """Tint the differing pixels red over the evidence and encode as PNG."""
    tinted = evidence_bgr.copy()
    tinted[mask > 0] = (0, 0, 255)  # BGR red
    overlaid = cv2.addWeighted(evidence_bgr, 1.0 - alpha, tinted, alpha, 0)
    ok, buffer = cv2.imencode(".png", overlaid)
    return buffer.tobytes() if ok else None
