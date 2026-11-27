"""Drop the `cases` and `case_items` tables (Cases was removed, 27 Nov 2026).

The app no longer reads or writes either table, and `ensure_schema` no longer
creates them. Custody entries are left alone: the log is never rewritten, and
custody.py keeps the CASE_ATTACHED / CASE_DETACHED labels so any older entry
still reads correctly.

Usage (from the backend/ directory, so .env is found):

    python scripts/drop_cases.py            # dry run: reports, changes nothing
    python scripts/drop_cases.py --apply    # drops both tables

Safe to re-run: both drops are guarded by IF EXISTS.
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


def count(cur, table: str) -> int | None:
    cur.execute("SELECT to_regclass(%s)", (table,))
    if cur.fetchone()[0] is None:
        return None
    cur.execute(f"SELECT count(*) FROM {table}")
    return cur.fetchone()[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--apply", action="store_true", help="drop the tables")
    args = parser.parse_args()

    conn = psycopg2.connect(get_database_url())
    try:
        with conn, conn.cursor() as cur:
            # case_items first: it holds the foreign key to cases.
            for table in ("case_items", "cases"):
                rows = count(cur, table)
                if rows is None:
                    print(f"{table}: already gone")
                    continue
                print(f"{table}: {rows} row(s)")
                if args.apply:
                    cur.execute(f"DROP TABLE IF EXISTS {table}")
                    print(f"{table}: dropped")
        if not args.apply:
            print("Dry run: nothing changed. Re-run with --apply to drop.")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
