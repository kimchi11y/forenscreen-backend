"""Find files in backend/uploads/ that no stored check refers to.

The uploads folder should hold exactly one file per row in `evidence_checks`,
plus the difference-map cache files built from them. Before the upload route
was fixed, files were written before the check was stored, so failed or
non-image uploads left copies behind that nothing points at. This script finds
them, and also reports the reverse problem: stored checks whose file is missing
or no longer matches its recorded SHA-256.

Usage (from the backend/ directory, so .env is found):

    python scripts/find_junk_uploads.py            # dry run: reports, changes nothing
    python scripts/find_junk_uploads.py --apply    # moves junk aside

Nothing is ever deleted. --apply MOVES junk into
backend/uploads_quarantine/<timestamp>/ together with a manifest.json saying
why each file was moved, so any mistake can be undone by moving files back.

Files referenced by a check are never moved, even when their hash no longer
matches: a mismatched file is evidence that something went wrong, and is only
reported.
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from datetime import datetime

import psycopg2
from dotenv import dotenv_values

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_DIR = os.path.join(BACKEND_DIR, "uploads")
QUARANTINE_ROOT = os.path.join(BACKEND_DIR, "uploads_quarantine")

# Files that belong in the folder and are not evidence.
KEEP = {".gitignore"}

# The JSON registry used by the old duplicate checker, which has been removed.
OBSOLETE = {"file_registry.json"}

# Difference-map cache: <evidence id>__<original id>__regions.json / __overlay.png
SIDECAR = re.compile(
    r"^(?P<file_id>[0-9a-f-]{36})__[0-9a-f-]{36}__(regions\.json|overlay\.png)$"
)


def get_database_url() -> str:
    env_path = os.path.join(BACKEND_DIR, ".env")
    url = dotenv_values(env_path).get("DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        sys.exit(f"No DATABASE_URL found in {env_path} or the environment.")
    return url


def sha256_of(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify(checks: dict) -> tuple[list, list, list]:
    """Return (junk, missing, mismatched).

    `checks` maps saved_filename -> (check id, recorded sha256).
    junk is [(filename, reason)]; missing and mismatched are [(check id, filename)].
    """
    check_ids = {check_id for check_id, _ in checks.values()}
    junk, mismatched = [], []

    for name in sorted(os.listdir(UPLOAD_DIR)):
        path = os.path.join(UPLOAD_DIR, name)
        if not os.path.isfile(path) or name in KEEP:
            continue

        if name in OBSOLETE:
            junk.append((name, "obsolete duplicate-checker registry"))
        elif name in checks:
            check_id, recorded = checks[name]
            if sha256_of(path) != recorded:
                mismatched.append((check_id, name))
        elif (match := SIDECAR.match(name)) is not None:
            if match["file_id"] not in check_ids:
                junk.append((name, "difference-map cache for a check that is not stored"))
        else:
            junk.append((name, "not referenced by any stored check"))

    missing = [
        (check_id, name)
        for name, (check_id, _) in sorted(checks.items())
        if not os.path.isfile(os.path.join(UPLOAD_DIR, name))
    ]
    return junk, missing, mismatched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="move junk into uploads_quarantine/ (default is a dry run)",
    )
    args = parser.parse_args()

    conn = psycopg2.connect(get_database_url(), connect_timeout=20)
    try:
        cur = conn.cursor()
        cur.execute("SELECT id::text, saved_filename, sha256_hash FROM evidence_checks;")
        checks = {name: (check_id, sha) for check_id, name, sha in cur.fetchall()}
    finally:
        conn.close()

    junk, missing, mismatched = classify(checks)
    print(f"evidence_checks: {len(checks)} rows")

    print(f"\njunk files: {len(junk)}")
    for name, reason in junk:
        print(f"  {name}  ({reason})")

    print(f"\nchecks whose file is missing: {len(missing)}")
    for check_id, name in missing:
        print(f"  check {check_id}  expected {name}")

    print(f"\nchecks whose file no longer matches its SHA-256: {len(mismatched)}")
    for check_id, name in mismatched:
        print(f"  check {check_id}  {name}")

    if not args.apply:
        print("\nDRY RUN - nothing moved. Re-run with --apply to move the junk files aside.")
        return 0

    if not junk:
        print("\nNothing to move.")
        return 0

    target = os.path.join(QUARANTINE_ROOT, datetime.now().strftime("%Y%m%d-%H%M%S"))
    os.makedirs(target)
    for name, _ in junk:
        shutil.move(os.path.join(UPLOAD_DIR, name), os.path.join(target, name))

    manifest = {
        "moved_at": datetime.now().isoformat(),
        "source": UPLOAD_DIR,
        "files": [{"name": name, "reason": reason} for name, reason in junk],
    }
    with open(os.path.join(target, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nmoved {len(junk)} file(s) to {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
