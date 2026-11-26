"""Render a check's report as a PDF a victim can hand to the police or a school.

The PDF is a presentation of the same report the JSON export returns - the
verdict, the fingerprints, the blockchain anchor of the matched original and
the full chain of custody - plus the evidence image and, where the verdict
compares against a different original, the difference heatmap.

It is written to be read by someone who has never heard of ForenScreen, so
every technical value is introduced in plain words, and it ends with how to
check the claims independently instead of trusting the document.
"""

import io
from datetime import datetime, timezone

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import (
    Image as PdfImage,
    KeepTogether,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from xml.sax.saxutils import escape

# Plain-language verdicts. Kept in step with the frontend's lib/verdict.js.
VERDICTS = {
    "VERIFIED_AUTHENTIC": (
        "Original, unmodified",
        "The file is byte-for-byte identical to a registered original. Nothing in it has changed.",
        "#15803d",
    ),
    "RESAVED_COPY": (
        "Re-saved copy, no edits found",
        "The picture matches a registered original and shows all of it. The file was re-saved or "
        "resized, which social media platforms do to every upload, but no part of the picture was "
        "found to differ.",
        "#1d4ed8",
    ),
    "TAMPERED_MANIPULATED": (
        "Modified copy detected",
        "The picture matches a registered original, but it has been changed: edited, cropped or "
        "otherwise altered. Where possible, the changed areas are marked in red below.",
        "#b91c1c",
    ),
    "EARLIER_ORIGINAL_EXISTS": (
        "Earlier similar original found",
        "This exact file was registered, but only after a similar image had already been "
        "registered by another account. It may be an edited copy of that earlier original.",
        "#b45309",
    ),
    "AMBIGUOUS_MATCH": (
        "Needs confirmation",
        "The image is close to more than one registered original, so it could not be tied to one.",
        "#b45309",
    ),
    "UNREGISTERED_ASSET": (
        "No match found",
        "No registered original matches this image. That is not a judgement about the picture.",
        "#525252",
    ),
    "PROCESSING_ERROR": ("Could not read file", "The file could not be decoded.", "#525252"),
}

MATCH_METHODS = {
    "sha256_exact": "Exact file match (SHA-256)",
    "orb_ransac": "Visual match (ORB feature points, RANSAC alignment)",
    "phash_fallback": "Visual match (perceptual hash, low-detail image)",
    "registered_after_similar": "Exact match to a registration made after an earlier similar original",
}

_styles = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=_styles["Title"], fontSize=18, leading=22, alignment=TA_LEFT, spaceAfter=2)
# keepWithNext stops a heading being left alone at the bottom of a page.
H2 = ParagraphStyle(
    "h2", parent=_styles["Heading2"], fontSize=12, leading=15, spaceBefore=10, spaceAfter=4,
    keepWithNext=1,
)
BODY = ParagraphStyle("body", parent=_styles["BodyText"], fontSize=9, leading=12)
SMALL = ParagraphStyle("small", parent=BODY, fontSize=7.5, leading=10, textColor=colors.HexColor("#525252"))
MONO = ParagraphStyle("mono", parent=BODY, fontName="Courier", fontSize=7.5, leading=9.5)
CELL = ParagraphStyle("cell", parent=BODY, fontSize=8, leading=10)


def _p(text, style=BODY):
    return Paragraph(escape(str(text)) if text is not None else "—", style)


def _when(value):
    if not value:
        return "—"
    try:
        moment = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)  # the database stores UTC
    return moment.astimezone(timezone.utc).strftime("%d %b %Y, %H:%M UTC")


def _fact_table(rows):
    table = Table(
        [[_p(label, CELL), value if not isinstance(value, str) else _p(value, CELL)] for label, value in rows],
        colWidths=[45 * mm, None],
    )
    table.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TEXTCOLOR", (0, 0), (0, -1), colors.HexColor("#525252")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e5e5")),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _image(data, max_w, max_h):
    """A reportlab image scaled to fit the box, or None if it cannot be read."""
    if not data:
        return None
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as probe:
            width, height = probe.size
            if probe.format not in ("PNG", "JPEG"):
                buffer = io.BytesIO()
                probe.convert("RGB").save(buffer, "JPEG", quality=85)
                data = buffer.getvalue()
    except Exception:
        return None
    scale = min(max_w / width, max_h / height)
    return PdfImage(io.BytesIO(data), width=width * scale, height=height * scale)


def render(report: dict, evidence_image: bytes | None, overlay_image: bytes | None,
           anchor: dict | None, contract_address: str | None) -> bytes:
    """The report as PDF bytes."""
    check = report["check"]
    matched = check.get("matched_asset") or {}
    label, explanation, colour = VERDICTS.get(
        check["verdict"], (check["verdict"], "", "#525252")
    )

    story = [
        Paragraph("ForenScreen verification report", H1),
        _p(f"Generated {_when(report['generated_at'])} · Check {check['id']}", SMALL),
        Spacer(1, 6 * mm),
    ]

    verdict_box = Table(
        [[Paragraph(f'<font color="{colour}"><b>{escape(label)}</b></font>', ParagraphStyle(
            "verdict", parent=BODY, fontSize=14, leading=18))],
         [_p(explanation)]],
        colWidths=[None],
    )
    verdict_box.setStyle(TableStyle([
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor(colour)),
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#fafafa")),
        ("LEFTPADDING", (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
    ]))
    story += [verdict_box, Spacer(1, 3 * mm)]

    story += [Paragraph("The checked file", H2), _fact_table([
        ("Checked at", _when(check.get("checked_at"))),
        ("SHA-256 fingerprint", _p(check.get("sha256"), MONO)),
        ("How it was decided", MATCH_METHODS.get(check.get("match_method"), check.get("match_method") or "—")),
        *([("Matching feature points", f"{check['orb_inliers']}"
            + (f" ({check['orb_inlier_ratio'] * 100:.0f}% geometrically consistent)"
               if check.get("orb_inlier_ratio") is not None else ""))]
          if check.get("orb_inliers") is not None else []),
    ])]

    if matched:
        story += [Paragraph("The registered original it was compared with", H2), _fact_table([
            ("Original", matched.get("file_name") or "Registered by another account (kept private)"),
            ("Registered at", _when(matched.get("registered_at"))),
            ("SHA-256 fingerprint", _p(matched.get("sha256"), MONO)),
        ])]

        story.append(Paragraph("Blockchain record of the original", H2))
        if anchor and anchor.get("anchored"):
            anchored_at = datetime.fromtimestamp(anchor["timestamp"], tz=timezone.utc)
            story.append(_fact_table([
                ("Status", "Anchored on the Ethereum Sepolia test network"),
                ("Anchored at", anchored_at.strftime("%d %b %Y, %H:%M UTC")),
                ("Contract", _p(contract_address, MONO)),
                ("Public record", _p(anchor.get("explorer"), MONO)),
            ]))
            story.append(_p(
                "The anchor proves this fingerprint existed by the time shown and has not been changed "
                "since. Nobody, including ForenScreen, can alter or remove it.", SMALL))
        elif anchor and anchor.get("available"):
            story.append(_p("The original's fingerprint is not anchored on the blockchain yet."))
        else:
            story.append(_p("The blockchain could not be reached when this report was made."))

    images = []
    evidence = _image(evidence_image, 80 * mm, 110 * mm)
    if evidence is not None:
        images.append([evidence, _p("The checked image", SMALL)])
    overlay = _image(overlay_image, 80 * mm, 110 * mm)
    if overlay is not None:
        images.append([overlay, _p("Red areas differ from the registered original", SMALL)])
    if images:
        grid = Table([[cell[0] for cell in images], [cell[1] for cell in images]])
        grid.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0)]))
        story += [Paragraph("Images", H2), KeepTogether([grid])]
    elif check.get("saved_filename"):
        story += [Paragraph("Images", H2),
                  _p("The stored copy of the checked image could not be included: it failed its integrity check.")]

    chain = report["custody"]["chain"]
    entries = report["custody"]["entries"]
    intro = _p(
        "Every action on this check is recorded in an append-only log. Each entry includes the "
        "fingerprint of the one before it, so changing or removing any entry breaks every later one."
    )
    intro.keepWithNext = 1
    story += [Paragraph("Chain of custody", H2), intro, Spacer(1, 2 * mm)]
    rows = [[_p("#", CELL), _p("When", CELL), _p("What happened", CELL), _p("By", CELL), _p("Entry hash", CELL)]]
    for entry in entries:
        actor = entry.get("actor") or ""
        rows.append([
            _p(entry.get("seq"), CELL), _p(_when(entry.get("occurred_at")), CELL),
            _p(entry.get("label"), CELL),
            _p("signed-in user" if actor.startswith("user:") else actor, CELL),
            _p((entry.get("entry_hash") or "")[:16] + "…", MONO),
        ])
    log = Table(rows, colWidths=[10 * mm, 38 * mm, 52 * mm, 28 * mm, None], repeatRows=1)
    log.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f5f5f5")),
        ("LINEBELOW", (0, 0), (-1, -1), 0.25, colors.HexColor("#e5e5e5")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(log)
    story.append(Spacer(1, 2 * mm))
    intact = chain.get("intact")
    story.append(Paragraph(
        f'<b>Chain check:</b> <font color="{"#15803d" if intact else "#b91c1c"}">'
        f'{"intact, no entry has been changed" if intact else "BROKEN, an entry has been changed"}</font>'
        + (f' · head {escape(chain["head_hash"][:16])}…' if chain.get("head_hash") else ""),
        BODY,
    ))

    story += [Paragraph("How to check this report yourself", H2), *[_p(step) for step in (
        "1. Compute the SHA-256 of the image file you were given (for example with "
        "\"shasum -a 256 file.png\") and compare it with the checked file's fingerprint above.",
        "2. Look up the original's fingerprint on the public blockchain record linked above, or on "
        "ForenScreen's /verify page, to see when it was registered.",
        "3. This report records the export itself in the chain of custody, so a later report will "
        "list this one.",
    )], Spacer(1, 3 * mm), _p(
        "Limits: ForenScreen proves which image was registered first and whether a copy differs from "
        "it. It does not prove who created the original, and metadata inside image files can be "
        "edited by anyone.", SMALL)]

    buffer = io.BytesIO()
    SimpleDocTemplate(
        buffer, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm,
        topMargin=16 * mm, bottomMargin=16 * mm,
        title=f"ForenScreen report {check['id']}", author="ForenScreen",
    ).build(story)
    return buffer.getvalue()
