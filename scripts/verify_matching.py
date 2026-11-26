"""Verification harness for the Phase-1 matching fixes.

Runs the real FastAPI app against the live Neon registry and asserts the
verdict ladder:

    exact re-upload      -> VERIFIED_AUTHENTIC   (checked before perceptual work)
    re-encoded original  -> RESAVED_COPY         (same picture, different bytes)
    anonymous register   -> refused, HTTP 401    (registering needs an account)
    unrelated image      -> UNREGISTERED_ASSET

Note: every check below is stored in the LIVE registry with a custody entry,
which can never be deleted. For a measured evaluation that leaves the live
data alone, use scripts/evaluate_accuracy.py instead.

Usage (from the backend/ directory):
    python scripts/verify_matching.py
"""

import io
import sys

import os
from PIL import Image
import psycopg2
import httpx
from dotenv import dotenv_values

BASE = os.environ.get("FORENSCREEN_BASE", "http://127.0.0.1:8000")
BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def database_url():
    url = dotenv_values(os.path.join(BACKEND_DIR, ".env")).get("DATABASE_URL")
    if not url:
        sys.exit("No DATABASE_URL in backend/.env")
    return url


def fetch_one_registered():
    conn = psycopg2.connect(database_url(), connect_timeout=20)
    cur = conn.cursor()
    cur.execute(
        "SELECT file_name, file_data, sha256_hash FROM original_assets "
        "ORDER BY created_at ASC LIMIT 1;"
    )
    row = cur.fetchone()
    row = (row[0], bytes(row[1]), row[2])  # BYTEA arrives as a memoryview
    cur.execute("SELECT count(*) FROM original_assets;")
    total = cur.fetchone()[0]
    conn.close()
    return row, total


def upload_evidence(name, data, mime="image/jpeg"):
    r = httpx.post(
        f"{BASE}/upload", files={"file": (name, data, mime)}, timeout=120
    )
    r.raise_for_status()
    return r.json()["files"][0]


def register(name, data, mime="image/jpeg"):
    r = httpx.post(
        f"{BASE}/upload-asset/", files={"file": (name, data, mime)}, timeout=120
    )
    return r.status_code, r.json()


def main():
    (file_name, file_data, sha256), total = fetch_one_registered()
    print(f"registry: {total} originals; using '{file_name}' as the reference")
    print(f"reference sha256: {sha256[:16]}...  ({len(file_data)} bytes)\n")

    results = []

    # --- 1. Exact byte match: must be VERIFIED_AUTHENTIC -------------------
    got = upload_evidence(file_name, file_data)["verification_status"]
    results.append(("exact re-upload", "VERIFIED_AUTHENTIC", got))

    # --- 2. Same picture, re-encoded: same pHash, different bytes ----------
    img = Image.open(io.BytesIO(file_data)).convert("RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    reencoded = buf.getvalue()
    got = upload_evidence("reencoded.jpg", reencoded)["verification_status"]
    results.append(("re-encoded original", "RESAVED_COPY", got))

    # --- 2b. Changed framing. pHash alone rejected these (12-14 bits, past the
    #        old threshold of 10); ORB should still recognise the picture. ---
    cropped = img.crop(
        (int(img.width * .05), int(img.height * .05),
         int(img.width * .95), int(img.height * .95))
    )
    buf = io.BytesIO()
    cropped.save(buf, format="JPEG", quality=90)
    got = upload_evidence("cropped.jpg", buf.getvalue())["verification_status"]
    results.append(("cropped 5% each side", "TAMPERED_MANIPULATED", got))

    barred = Image.new("RGB", (img.width, int(img.height * 1.08)), "white")
    barred.paste(img, (0, 0))
    buf = io.BytesIO()
    barred.save(buf, format="JPEG", quality=90)
    got = upload_evidence("with-bar.jpg", buf.getvalue())["verification_status"]
    results.append(("+ 8% platform UI bar", "TAMPERED_MANIPULATED", got))

    # --- 3. Registering without signing in: refused --------------------
    status, _ = register(file_name, file_data)
    results.append(("anonymous register", "HTTP 401", f"HTTP {status}"))

    # --- 4. Unrelated image: no candidate within threshold -----------------
    noise = Image.frombytes("RGB", (256, 256), os.urandom(256 * 256 * 3))
    buf = io.BytesIO()
    noise.save(buf, format="PNG")
    got = upload_evidence("unrelated.png", buf.getvalue(), "image/png")[
        "verification_status"
    ]
    results.append(("unrelated image", "UNREGISTERED_ASSET", got))

    print(f"{'case':<24} {'expected':<26} {'actual':<26} result")
    ok = True
    for case, exp, act in results:
        good = exp == act
        ok &= good
        print(f"{case:<24} {exp:<26} {act:<26} {'PASS' if good else 'FAIL'}")

    print("\n" + ("ALL PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
