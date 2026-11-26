"""Metadata recorded when an original is registered, each value labelled by source.

Why this exists
---------------
A registration says "this file existed by this time". Metadata adds context a
reviewer can weigh: when the picture claims to have been taken, on what device,
with which software, and what the registering browser reported. None of it is
proof - EXIF is trivially editable - so every value carries the name of where
it came from, and the strength of the record still rests on the SHA-256 and its
on-chain anchor.

Sources, from most to least trustworthy:

    server   - this server's own clock; the client cannot influence it
    file     - read from the image's EXIF / XMP / PNG text; set by whatever
               produced the file, and editable by anyone
    device   - reported by the registering browser; self-reported
    location - the browser's geolocation, only when the user switched it on

What screenshots carry
----------------------
Very little. Measured on real registrations, most screenshots had no EXIF at
all; an iPhone screenshot carried a "Screenshot" user comment and XMP. Camera
fields matter mostly for photos. Editing software, though, often leaves a
trace even in a screenshot (XMP CreatorTool, PNG "Software"), which is what
the signals below look for.

Privacy
-------
GPS coordinates found inside the file are only kept when the user has switched
"Share my location" on; otherwise only the fact that the file contained GPS is
recorded. Coordinates are never written to the custody log, which is
append-only and could never be cleared of them - the log stores a hash of the
metadata instead.
"""

import hashlib
import io
import json
import re
from datetime import datetime, timezone

from PIL import Image, ExifTags

EXIF_IFD = 0x8769
GPS_IFD = 0x8825

# Substrings that identify editing software in Software / CreatorTool fields.
# Matched case-insensitively. Camera firmware and OS screenshot tools are not
# here: their presence says nothing about editing.
EDITING_SOFTWARE = (
    "photoshop", "lightroom", "gimp", "snapseed", "picsart", "canva", "pixlr",
    "affinity", "facetune", "fotor", "photopea", "paint.net", "pixelmator",
    "inshot", "vsco", "meitu", "illustrator", "figma",
)

# Device fields accepted from the browser, with a length cap each, so a client
# cannot store arbitrary data under the registration.
DEVICE_FIELDS = {
    "user_agent": 400,
    "platform": 100,
    "language": 35,
    "timezone": 64,
    "screen": 32,
}

_XMP_FIELDS = {
    "creator_tool": r'(?:xmp:CreatorTool="([^"]*)"|<xmp:CreatorTool>([^<]*)<)',
    "created_at": r'(?:xmp:CreateDate="([^"]*)"|<xmp:CreateDate>([^<]*)<)',
    "modified_at": r'(?:xmp:ModifyDate="([^"]*)"|<xmp:ModifyDate>([^<]*)<)',
    "user_comment": r"<exif:UserComment>.*?<rdf:li[^>]*>([^<]*)<",
}


def _clean(value, limit=200):
    """Normalise a metadata value to a short printable string, or None."""
    if value is None:
        return None
    if isinstance(value, bytes):
        # EXIF UserComment starts with an 8-byte character-code header.
        value = value[8:] if value[:8] in (b"ASCII\0\0\0", b"UNICODE\0", b"\0" * 8) else value
        value = value.decode("utf-8", "ignore")
    text = "".join(ch for ch in str(value) if ch.isprintable()).strip()
    return text[:limit] or None


def _exif_datetime(value):
    """EXIF "YYYY:MM:DD HH:MM:SS" (no zone) -> "YYYY-MM-DDTHH:MM:SS", or None."""
    text = _clean(value, 40)
    if not text:
        return None
    try:
        return datetime.strptime(text[:19], "%Y:%m:%d %H:%M:%S").isoformat()
    except ValueError:
        return None


def _gps_decimal(gps):
    """GPS IFD -> (lat, lon) in decimal degrees, or None."""
    try:
        def degrees(dms, ref):
            d, m, s = (float(x) for x in dms)
            value = d + m / 60 + s / 3600
            return -value if ref in ("S", "W") else value

        lat = degrees(gps[2], gps.get(1))
        lon = degrees(gps[4], gps.get(3))
    except (KeyError, TypeError, ValueError, ZeroDivisionError, IndexError):
        return None
    return round(lat, 6), round(lon, 6)


def read_file_metadata(image_bytes: bytes) -> dict:
    """Everything the file itself says about its origin. Values may be None."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
    except Exception:  # unreadable files were refused before this is called
        return {"source": "file"}

    exif = image.getexif()
    sub = exif.get_ifd(EXIF_IFD)
    gps = exif.get_ifd(GPS_IFD)
    tags = {ExifTags.TAGS.get(k, k): v for k, v in {**exif, **sub}.items()}

    xmp_raw = image.info.get("xmp") or image.info.get("XML:com.adobe.xmp") or b""
    xmp = xmp_raw.decode("utf-8", "ignore") if isinstance(xmp_raw, bytes) else str(xmp_raw)
    from_xmp = {}
    for field, pattern in _XMP_FIELDS.items():
        match = re.search(pattern, xmp, re.S)
        if match:
            from_xmp[field] = _clean(next(g for g in match.groups() if g is not None))

    result = {
        "source": "file",
        "format": image.format,
        "width": image.width,
        "height": image.height,
        "captured_at": _exif_datetime(tags.get("DateTimeOriginal")) or from_xmp.get("created_at"),
        "modified_at": _exif_datetime(tags.get("DateTime")) or from_xmp.get("modified_at"),
        "camera_make": _clean(tags.get("Make"), 80),
        "camera_model": _clean(tags.get("Model"), 80),
        "software": _clean(tags.get("Software"), 120)
        or from_xmp.get("creator_tool")
        or _clean(image.info.get("Software"), 120),
        "user_comment": _clean(tags.get("UserComment"), 120) or from_xmp.get("user_comment"),
        "gps_present": bool(gps),
    }
    coordinates = _gps_decimal(gps) if gps else None
    if coordinates:
        result["_gps"] = coordinates  # dropped later unless the user shared location
    return result


def clean_device(raw) -> dict | None:
    """Keep only the expected browser fields, each length-capped."""
    if not isinstance(raw, dict):
        return None
    device = {
        field: _clean(raw.get(field), limit)
        for field, limit in DEVICE_FIELDS.items()
        if _clean(raw.get(field), limit)
    }
    return {"source": "device (reported by the browser)", **device} if device else None


def clean_location(raw) -> dict | None:
    """A browser geolocation fix, validated, or None."""
    if not isinstance(raw, dict):
        return None
    try:
        lat, lon = float(raw["lat"]), float(raw["lon"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        return None
    accuracy = raw.get("accuracy_m")
    return {
        "source": "location (browser geolocation, shared by the user)",
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "accuracy_m": round(float(accuracy)) if isinstance(accuracy, (int, float)) else None,
    }


def signals(file_meta: dict, received_at: datetime) -> list[dict]:
    """Things in the metadata a reviewer should look at. Hints, not verdicts."""
    found = []
    software = (file_meta.get("software") or "").lower()
    tool = next((name for name in EDITING_SOFTWARE if name in software), None)
    if tool:
        found.append({
            "code": "EDITING_SOFTWARE",
            "message": f"The file says it was last saved by editing software ({file_meta['software']}).",
        })

    captured, modified = file_meta.get("captured_at"), file_meta.get("modified_at")
    if captured and modified and modified[:19] > captured[:19]:
        found.append({
            "code": "MODIFIED_AFTER_CAPTURE",
            "message": f"The file was modified ({modified[:16]}) after it was taken ({captured[:16]}).",
        })

    # EXIF times have no zone, so allow a day of slack before calling it future.
    if captured:
        try:
            taken = datetime.fromisoformat(captured[:19]).replace(tzinfo=timezone.utc)
            if (taken - received_at).total_seconds() > 86400:
                found.append({
                    "code": "CAPTURED_IN_FUTURE",
                    "message": "The file claims to have been taken after it was received.",
                })
        except ValueError:
            pass
    return found


def build(image_bytes: bytes, client: dict | None, received_at: datetime) -> dict:
    """The full registration metadata record, every section labelled by source."""
    client = client if isinstance(client, dict) else {}
    file_meta = read_file_metadata(image_bytes)
    location = clean_location(client.get("location"))

    # GPS inside the file is personal data too: keep it only when the user
    # chose to share location, otherwise record just that it was there.
    file_gps = file_meta.pop("_gps", None)
    if file_gps and location is not None:
        file_meta["gps"] = {"lat": file_gps[0], "lon": file_gps[1]}

    return {
        "server": {"source": "server", "received_at": received_at.isoformat()},
        "file": file_meta,
        "device": clean_device(client.get("device")),
        "location": location,
        "signals": signals(file_meta, received_at),
    }


def for_custody(record: dict) -> dict:
    """What goes into the append-only custody log: no coordinates, plus a hash.

    The hash commits the log to the full record, so the stored metadata cannot
    be altered unnoticed, while the coordinates themselves stay deletable.
    """
    canonical = json.dumps(record, sort_keys=True, separators=(",", ":"), default=str)
    file_meta = {k: v for k, v in (record.get("file") or {}).items() if k != "gps"}
    return {
        "metadata_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
        "received_at": record["server"]["received_at"],
        "file": file_meta,
        "device": record.get("device"),
        "location_shared": record.get("location") is not None,
        "signals": [s["code"] for s in record.get("signals", [])],
    }
