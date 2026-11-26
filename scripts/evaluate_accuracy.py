"""Measure how accurately ForenScreen classifies copies of real screenshots.

This is the evaluation for FYP objective 5. It runs the real application - the
same /upload-asset and /upload routes the web app calls - over a folder of
screenshots, and records every verdict.

Each screenshot is registered under its own test account. Then, for each one:

    exact copy            -> expected "Original"           (VERIFIED_AUTHENTIC)
    platform re-saves     -> expected "Re-saved, no edits" (RESAVED_COPY)
    edits and crops       -> expected "Modified"           (TAMPERED_MANIPULATED)

Registering every screenshot also measures false alarms: the screenshots are
all different pictures, so any "a similar image is already registered" warning
between two of them is an ORB false positive (to-do A4 item 9).

Edits are generated (a box over text, a sentence, a number, one digit), drawn
in a colour that contrasts with the local background so every edit really
changes the picture. Real edited copies can be added with --manual.

Nothing touches the live registry. Everything runs in a temporary Postgres
schema that is dropped at the end, anchoring is switched off, and uploads go
to a temporary folder.

Usage (from the backend/ directory):

    python scripts/evaluate_accuracy.py SCREENSHOTS_DIR [--out RESULTS_DIR]
                                        [--manual MANUAL_DIR]

SCREENSHOTS_DIR holds the originals (PNG/JPEG/HEIC/WebP). Use screenshots that
were NOT used to tune the thresholds - the 5 registered on 26 Nov 2026 were.

MANUAL_DIR (optional) holds real copies you made yourself, named
    <original file stem>__<expected>__<anything>.<ext>
where <expected> is one of: original, resaved, modified.
For example "chat1__modified__photoshop.png" or "chat1__resaved__whatsapp.jpg".

Writes results.csv (one row per comparison) and results.md (summary tables
for the report) to RESULTS_DIR (default: ./evaluation-results).
"""

import argparse
import csv
import io
import os
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlsplit, urlunsplit

import psycopg2
from dotenv import dotenv_values
from PIL import Image, ImageDraw, ImageStat
from pillow_heif import register_heif_opener

BACKEND_DIR = Path(__file__).resolve().parent.parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

register_heif_opener()

SCHEMA = "evaluation_tmp"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".heic", ".heif", ".webp"}

EXPECTED = {
    "original": "VERIFIED_AUTHENTIC",
    "resaved": "RESAVED_COPY",
    "modified": "TAMPERED_MANIPULATED",
}
LABELS = {
    "VERIFIED_AUTHENTIC": "Original",
    "RESAVED_COPY": "Re-saved, no edits",
    "TAMPERED_MANIPULATED": "Modified",
    "EARLIER_ORIGINAL_EXISTS": "Earlier similar original",
    "AMBIGUOUS_MATCH": "Needs confirmation",
    "UNREGISTERED_ASSET": "No match",
    "PROCESSING_ERROR": "Could not read",
}


# --- generated copies ---------------------------------------------------------

def _encode(img, fmt="PNG", quality=None, scale=1.0):
    img = img.convert("RGB")
    if scale != 1.0:
        img = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))), Image.LANCZOS)
    buffer = io.BytesIO()
    img.save(buffer, fmt, **({"quality": quality} if quality else {}))
    return buffer.getvalue()


def _ink(img, box):
    """Black or white, whichever stands out from the area being written on."""
    mean = ImageStat.Stat(img.convert("L").crop(box)).mean[0]
    return "white" if mean < 128 else "black"


def _write(img, text, x, y, size):
    copy = img.copy()
    left, top = int(img.width * x), int(img.height * y)
    box = (left, top, min(img.width, left + len(text) * size), min(img.height, top + size))
    ImageDraw.Draw(copy).text((left, top), text, fill=_ink(img, box), font_size=size)
    return copy


def generated_copies(img):
    """(case, kind, expected, bytes) for every generated copy of one screenshot."""
    img = img.convert("RGB")
    w, h = img.size
    boxed = img.copy()
    ImageDraw.Draw(boxed).rectangle([w * .15, h * .40, w * .85, h * .47], fill="white")
    number = _write(img, "500", .5, .3, max(14, w // 40))
    return [
        ("exact copy", "exact", "original", None),
        ("platform: re-saved JPEG q85", "platform", "resaved", _encode(img, "JPEG", 85)),
        ("platform: 75% size, JPEG q80", "platform", "resaved", _encode(img, "JPEG", 80, .75)),
        ("platform: 50% size, JPEG q60", "platform", "resaved", _encode(img, "JPEG", 60, .5)),
        ("platform: 40% size, JPEG q50", "platform", "resaved", _encode(img, "JPEG", 50, .4)),
        ("edit: box over a line of text", "edit", "modified", _encode(boxed)),
        ("edit: sentence added", "edit", "modified",
         _encode(_write(img, "I hate all of you", .2, .55, max(18, w // 25)))),
        ("edit: number '500' added", "edit", "modified", _encode(number)),
        ("edit: one digit added", "edit", "modified", _encode(_write(img, "1", .3, .7, max(12, w // 60)))),
        ("edit: '500', then platform 75% q80", "edit", "modified", _encode(number, "JPEG", 80, .75)),
        ("crop: bottom 20% removed", "crop", "modified", _encode(img.crop((0, 0, w, int(h * .8))))),
    ]


def manual_copies(manual_dir, stems):
    """Real copies the user made, parsed from their file names."""
    found = []
    for path in sorted(Path(manual_dir).iterdir()):
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        parts = path.stem.split("__")
        if len(parts) < 2 or parts[0] not in stems or parts[1] not in EXPECTED:
            print(f"  skipping {path.name}: expected <stem>__<original|resaved|modified>__... ", file=sys.stderr)
            continue
        found.append((parts[0], f"manual: {path.name}", "manual", parts[1], path.read_bytes()))
    return found


# --- isolated application -----------------------------------------------------

def isolated_database_url(real_url):
    """The same database, connected directly (not via the pooler) with
    search_path set to the temporary schema, so the app only ever sees it."""
    parts = urlsplit(real_url)
    netloc = parts.netloc.replace("-pooler", "")
    query = (parts.query + "&" if parts.query else "") + f"options=-csearch_path%3D{SCHEMA}"
    return urlunsplit(parts._replace(netloc=netloc, query=query))


def load_app(upload_dir):
    """Import the app pointed at the temporary schema, with test-only overrides."""
    from fastapi.testclient import TestClient

    import auth
    import main

    if main.SessionLocal().execute(main.text("SELECT current_schema()")).scalar() != SCHEMA:
        sys.exit("Refusing to run: the app is not connected to the temporary schema.")

    main.UPLOAD_DIR = Path(upload_dir)
    main._anchor_asset = lambda *args, **kwargs: {"status": "skipped"}  # no Sepolia
    current = {"user": None}
    main.app.dependency_overrides[auth.optional_user] = lambda: current["user"]
    return TestClient(main.app), current


# --- the run ------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("screenshots", help="folder of original screenshots")
    parser.add_argument("--manual", help="folder of real copies, named <stem>__<expected>__<any>.<ext>")
    parser.add_argument("--out", default="evaluation-results", help="where to write results.csv / results.md")
    args = parser.parse_args()

    originals = sorted(p for p in Path(args.screenshots).iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if len(originals) < 2:
        sys.exit("Need at least 2 screenshots.")

    real_url = dotenv_values(BACKEND_DIR / ".env").get("DATABASE_URL") or os.environ.get("DATABASE_URL")
    if not real_url:
        sys.exit("No DATABASE_URL in backend/.env.")

    admin = psycopg2.connect(real_url, connect_timeout=30)
    admin.autocommit = True
    admin.cursor().execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE; CREATE SCHEMA {SCHEMA};")
    os.environ["DATABASE_URL"] = isolated_database_url(real_url)

    rows, false_alarms, not_registered = [], [], []
    try:
        with tempfile.TemporaryDirectory() as upload_dir:
            client, current = load_app(upload_dir)

            # 1. Register every screenshot, each under its own account. Any
            #    similarity warning here is between two different pictures.
            registered = {}
            print(f"Registering {len(originals)} screenshots...")
            for index, path in enumerate(originals):
                current["user"] = {"sub": f"evaluation-{index}"}
                data = path.read_bytes()
                response = client.post("/upload-asset/", files={"file": (path.name, data, "application/octet-stream")})
                if response.status_code == 409 and response.json().get("similar"):
                    false_alarms.append(path.name)
                    response = client.post(
                        "/upload-asset/",
                        files={"file": (path.name, data, "application/octet-stream")},
                        data={"confirm_similar": "true"},
                    )
                if response.status_code not in (200, 201):
                    print(f"  could not register {path.name}: HTTP {response.status_code} {response.text[:120]}")
                    not_registered.append(f"{path.name} (HTTP {response.status_code})")
                    continue
                registered[path.stem] = (response.json()["id"], data)

            # 2. Check every copy anonymously, as the public Check page does.
            current["user"] = None
            work = []
            for path in originals:
                if path.stem not in registered:
                    continue
                with Image.open(path) as img:
                    img.load()
                    for case, kind, expected, data in generated_copies(img):
                        work.append((path.stem, case, kind, expected, data or registered[path.stem][1]))
            if args.manual:
                work.extend(manual_copies(args.manual, set(registered)))

            print(f"Checking {len(work)} copies...")
            for number, (stem, case, kind, expected, data) in enumerate(work, 1):
                started = time.time()
                result = client.post("/upload", files={"file": ("copy", data, "application/octet-stream")})
                seconds = time.time() - started
                file_result = result.json()["files"][0] if result.status_code == 200 else {}
                verdict = file_result.get("verification_status", f"HTTP {result.status_code}")
                matched = (file_result.get("matched_asset") or {}).get("original_asset_id")
                # Correct means the expected verdict AND against the right
                # original - "Modified" pinned on the wrong screenshot is wrong.
                right_original = matched == registered[stem][0]
                correct = verdict == EXPECTED[expected] and right_original
                rows.append({
                    "screenshot": stem, "case": case, "kind": kind,
                    "expected": LABELS[EXPECTED[expected]], "got": LABELS.get(verdict, verdict),
                    "correct": correct, "matched_right_original": right_original,
                    "largest_change_px": (file_result.get("matched_asset") or {}).get("largest_change_px"),
                    "seconds": round(seconds, 2),
                })
                if number % 10 == 0 or number == len(work):
                    print(f"  {number}/{len(work)}")
    finally:
        try:
            import database
            database.engine.dispose()
        except Exception:
            pass
        admin.cursor().execute(f"DROP SCHEMA IF EXISTS {SCHEMA} CASCADE;")
        admin.close()

    write_results(Path(args.out), rows, false_alarms, len(registered), not_registered)
    return 0


def write_results(out, rows, false_alarms, registered, not_registered):
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "results.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    by_kind, by_case = defaultdict(lambda: [0, 0]), defaultdict(lambda: [0, 0])
    confusion = defaultdict(lambda: defaultdict(int))
    for row in rows:
        by_kind[row["kind"]][0] += row["correct"]
        by_kind[row["kind"]][1] += 1
        by_case[row["case"]][0] += row["correct"]
        by_case[row["case"]][1] += 1
        confusion[row["expected"]][row["got"]] += 1

    edits = [r for r in rows if r["expected"] == "Modified"]
    missed = sum(1 for r in edits if r["got"] in ("Original", "Re-saved, no edits"))
    pairs = registered * (registered - 1) // 2
    seconds = [r["seconds"] for r in rows]
    columns = sorted({r["got"] for r in rows})

    lines = [
        "# ForenScreen accuracy evaluation", "",
        f"{registered} screenshots, {len(rows)} copies checked.", "",
        *([f"**WARNING: {len(not_registered)} screenshot(s) could not be registered and were left out: "
           f"{', '.join(not_registered)}. Fix this before using the results.**", ""] if not_registered else []),
        "## Correct verdicts by type of copy", "",
        "| Copy type | Correct | Accuracy |", "|---|---|---|",
        *[f"| {kind} | {c}/{t} | {100 * c / t:.1f}% |" for kind, (c, t) in by_kind.items()],
        "", "## The two kinds of mistake", "",
        f"- **Dangerous**: an edited or cropped copy reported as unedited: "
        f"**{missed}/{len(edits)}** ({100 * missed / max(1, len(edits)):.1f}%)",
        f"- **Safe**: an unedited platform copy reported as modified: "
        f"**{sum(1 for r in rows if r['kind'] == 'platform' and r['got'] == 'Modified')}"
        f"/{sum(1 for r in rows if r['kind'] == 'platform')}**",
        f"- **False alarms at registration** (two different screenshots flagged as similar): "
        f"**{len(false_alarms)}** of {registered} registrations"
        + (f" ({', '.join(false_alarms)})" if false_alarms else "")
        + f", across {pairs} possible pairs. Check each flagged pair by eye: if the two really are "
        "the same picture (e.g. one is an edited copy of the other), it is a correct detection, and "
        "the folder should only contain unrelated screenshots",
        "", "## Confusion matrix (rows: expected, columns: verdict)", "",
        "| Expected \\ Got | " + " | ".join(columns) + " |",
        "|---" * (len(columns) + 1) + "|",
        *[f"| {exp} | " + " | ".join(str(confusion[exp].get(col, 0)) for col in columns) + " |"
          for exp in sorted(confusion)],
        "", "## By case", "",
        "| Case | Correct |", "|---|---|",
        *[f"| {case} | {c}/{t} |" for case, (c, t) in by_case.items()],
        "", f"Average time per check: {sum(seconds) / len(seconds):.2f} s (max {max(seconds):.2f} s).", "",
    ]
    (out / "results.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))
    print(f"Written to {out / 'results.csv'} and {out / 'results.md'}")


if __name__ == "__main__":
    raise SystemExit(main())
