"""Idempotent schema migration for ForenScreen (testing/orb-matching).

Applies three changes to `original_assets`, in this order:

  1. De-duplicate rows by `sha256_hash` (keeps the earliest row of each group).
     Two byte-identical registrations are what push the matcher into
     AMBIGUOUS_MATCH today, because the pHash distance between them is 0 and
     the ambiguity margin is 4.
  2. Create a btree index on `sha256_hash` so the new exact-match lookup
     (checked BEFORE any perceptual matching) is an index probe, not a scan.
  3. Create a UNIQUE index on `sha256_hash` so the database itself refuses a
     duplicate registration, and `ON CONFLICT` can be used by the endpoint.

Usage (from the backend/ directory, so .env is found):

    python scripts/migrate.py            # dry run: reports, changes nothing
    python scripts/migrate.py --apply    # performs the changes

Safe to re-run: every statement is guarded by IF NOT EXISTS, and the
de-duplication step is a no-op once no duplicate hashes remain.
"""

import argparse
import os
import sys

import psycopg2
from dotenv import dotenv_values

BACKEND_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def get_database_url() -> str:
    env_path = os.path.join(BACKEND_DIR, ".env")
    url = dotenv_values(env_path).get("DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not url:
        sys.exit(f"No DATABASE_URL found in {env_path} or the environment.")
    return url


def find_duplicates(cur):
    """Return [(sha256, [duplicate_ids...]), ...] - ids exclude the kept row."""
    cur.execute(
        """
        SELECT sha256_hash,
               (array_agg(id::text ORDER BY created_at ASC, id ASC))[2:] AS doomed
        FROM original_assets
        GROUP BY sha256_hash
        HAVING count(*) > 1
        ORDER BY sha256_hash;
        """
    )
    return [(row[0], list(row[1])) for row in cur.fetchall()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="perform the changes (default is a dry run)",
    )
    args = parser.parse_args()

    conn = psycopg2.connect(get_database_url(), connect_timeout=20)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("SELECT count(*), count(DISTINCT sha256_hash) FROM original_assets;")
    total, distinct = cur.fetchone()
    print(f"original_assets: {total} rows, {distinct} distinct sha256_hash")

    duplicates = find_duplicates(cur)
    doomed_total = sum(len(ids) for _, ids in duplicates)
    if duplicates:
        print(f"\nduplicate groups: {len(duplicates)} ({doomed_total} rows to remove)")
        for sha, ids in duplicates:
            print(f"  {sha[:16]}...  keep earliest, remove {len(ids)}: "
                  f"{', '.join(str(i)[:8] for i in ids)}")
    else:
        print("\nduplicate groups: none")

    if not args.apply:
        print("\nDRY RUN - nothing changed. Re-run with --apply to commit.")
        conn.close()
        return 0

    if doomed_total:
        cur.execute(
            "DELETE FROM original_assets WHERE id = ANY(%s::uuid[]);",
            ([i for _, ids in duplicates for i in ids],),
        )
        print(f"\ndeleted {cur.rowcount} duplicate row(s)")

    cur.execute(
        "CREATE INDEX IF NOT EXISTS idx_original_assets_sha256 "
        "ON original_assets (sha256_hash);"
    )
    print("index idx_original_assets_sha256: ensured")

    cur.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_original_assets_sha256 "
        "ON original_assets (sha256_hash);"
    )
    print("index uq_original_assets_sha256: ensured")

    conn.commit()

    cur.execute("SELECT count(*), count(DISTINCT sha256_hash) FROM original_assets;")
    print(f"\nafter: {cur.fetchone()} (rows, distinct hashes)")
    cur.execute(
        "SELECT indexname FROM pg_indexes WHERE tablename = 'original_assets' "
        "ORDER BY indexname;"
    )
    print("indexes:", [r[0] for r in cur.fetchall()])
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
