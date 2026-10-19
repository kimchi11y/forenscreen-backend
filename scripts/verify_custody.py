"""Verify B0 (persisted checks) and B1 (hash-chained custody) end to end."""
import glob
import io
import os
import sys

import httpx
import psycopg2
from PIL import Image, ImageDraw
from dotenv import dotenv_values

sys.path.insert(0, "/Users/hakim/Workspace/Forenscreen/backend")
import custody  # noqa: E402

BASE = "http://127.0.0.1:8000"
BACKEND = "/Users/hakim/Workspace/Forenscreen/backend"
DB = dotenv_values(os.path.join(BACKEND, ".env"))["DATABASE_URL"]

passed = []


def check(label, condition, detail=""):
    passed.append(condition)
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")


def jpeg(image, quality=92):
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


base = Image.open(sorted(glob.glob(os.path.join(BACKEND, "uploads", "*.jpeg")))[0]).convert("RGB")
w, h = base.size

variant_a = base.copy()
ImageDraw.Draw(variant_a).ellipse([int(w*.1), int(h*.1), int(w*.2), int(h*.2)], fill=(10, 200, 90))

variant_b = variant_a.copy()
ImageDraw.Draw(variant_b).rectangle([int(w*.55), int(h*.6), int(w*.8), int(h*.72)], fill=(230, 40, 40))

print("\n1. Register an original (creates ASSET_REGISTERED)")
r = httpx.post(f"{BASE}/upload-asset/", files={"file": ("b0-b1-asset.jpg", jpeg(variant_a), "image/jpeg")}, timeout=180)
r.raise_for_status()
asset = r.json()
asset_id = asset["id"]
print(f"   asset {asset_id[:8]}  duplicate={asset.get('duplicate')}")

r = httpx.get(f"{BASE}/custody/asset/{asset_id}", timeout=60)
r.raise_for_status()
entries = r.json()["entries"]
check("custody has ASSET_REGISTERED", len(entries) == 1 and entries[0]["event_type"] == "ASSET_REGISTERED",
      f"({len(entries)} entries)")
check("entry hash is 64 hex chars", len(entries[0]["entry_hash"]) == 64)
check("first entry chains from genesis", entries[0]["prev_hash"] in (None, custody.GENESIS_HASH))

print("\n2. Upload modified evidence (creates CHECK_PERFORMED)")
r = httpx.post(f"{BASE}/upload", files={"file": ("b0-b1-evidence.jpg", jpeg(variant_b), "image/jpeg")}, timeout=180)
r.raise_for_status()
f = r.json()["files"][0]
file_id = f["file_id"]
verdict = f["verification_status"]
print(f"   verdict={verdict}  persisted={f.get('persisted')}  method={(f.get('matched_asset') or {}).get('match_method')}")
check("verdict is TAMPERED_MANIPULATED", verdict == "TAMPERED_MANIPULATED", verdict)
check("check was persisted", f.get("persisted") is True)

print("\n3. The check survives independently of the request")
r = httpx.get(f"{BASE}/checks/{file_id}", timeout=60)
check("GET /checks/{id} returns it", r.status_code == 200 and r.json()["verification_status"] == verdict,
      f"HTTP {r.status_code}")
r = httpx.get(f"{BASE}/checks?limit=5", timeout=60)
check("GET /checks lists it", any(c["file_id"] == file_id for c in r.json()["checks"]))

print("\n4. Custody chain for the evidence")
r = httpx.get(f"{BASE}/custody/evidence/{file_id}", timeout=60)
ev = r.json()["entries"]
check("evidence has CHECK_PERFORMED", len(ev) == 1 and ev[0]["event_type"] == "CHECK_PERFORMED",
      f"({len(ev)} entries)")
check("payload records the verdict", ev[0]["payload"].get("verdict") == verdict)
check("actor recorded honestly", ev[0]["actor"] == "unauthenticated", ev[0]["actor"])

r = httpx.get(f"{BASE}/custody/evidence/{file_id}/verify", timeout=60)
rep = r.json()
check("chain verifies intact", rep["intact"] is True, f"head={str(rep.get('head_hash'))[:12]}")

print("\n5. Append-only enforcement (the database refuses to rewrite history)")
conn = psycopg2.connect(DB, connect_timeout=20)
conn.autocommit = False
cur = conn.cursor()
try:
    cur.execute("UPDATE custody_entries SET payload = '{}'::jsonb WHERE subject_id = %s", (file_id,))
    check("plain UPDATE is refused", False, "it succeeded - trigger missing")
    conn.rollback()
except psycopg2.Error as exc:
    check("plain UPDATE is refused", "append-only" in str(exc), str(exc).splitlines()[0][:70])
    conn.rollback()

try:
    cur.execute("DELETE FROM custody_entries WHERE subject_id = %s", (file_id,))
    check("DELETE is refused", False, "it succeeded - trigger missing")
    conn.rollback()
except psycopg2.Error as exc:
    check("DELETE is refused", "append-only" in str(exc), str(exc).splitlines()[0][:70])
    conn.rollback()

print("\n6. A tampered chain is detected (trigger disabled inside a rolled-back tx)")
cur.execute("ALTER TABLE custody_entries DISABLE TRIGGER custody_entries_append_only")
cur.execute(
    "UPDATE custody_entries SET payload = jsonb_set(payload, '{verdict}', '\"VERIFIED_AUTHENTIC\"') "
    "WHERE subject_id = %s",
    (file_id,),
)
cur.execute("SELECT seq, id, subject_type, subject_id, event_type, actor, occurred_at, payload, "
            "prev_hash, entry_hash FROM custody_entries WHERE subject_id = %s ORDER BY seq", (file_id,))
rows = cur.fetchall()
tampered = [
    {
        "seq": r[0], "id": str(r[1]), "subject_type": r[2], "subject_id": str(r[3]),
        "event_type": r[4], "actor": r[5], "occurred_at": r[6].isoformat(),
        "payload": r[7], "prev_hash": r[8], "entry_hash": r[9],
    }
    for r in rows
]
report = custody.verify_chain(tampered)
check("tampering is detected", report["intact"] is False,
      f"{len(report['problems'])} problem(s): {[p['problem'] for p in report['problems']]}")
conn.rollback()
conn.close()

print("\n" + ("ALL PASS" if all(passed) else f"{passed.count(False)} CHECK(S) FAILED"))
sys.exit(0 if all(passed) else 1)
