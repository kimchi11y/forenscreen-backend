"""Append-only, hash-chained chain of custody.

Chain of custody is the documented history of an item of evidence: who handled
it, when, and what they did with it. Two properties make it a *chain* rather
than a list, and both are enforced here:

1. **Append-only.** Nothing is updated or deleted. The trigger in `schema.sql`
   enforces this in the database, so an accidental `UPDATE` fails loudly instead
   of quietly destroying the audit trail.

2. **Each entry commits to its predecessor:**

       entry_hash = SHA-256( canonical_json(entry) + prev_hash )

   Rewriting any entry invalidates its own hash and every hash after it, so
   tampering is detectable by anyone who can read the table.

What this does *not* prove
--------------------------
It proves the log has not been altered by anyone who cannot also recompute the
hashes. The table owner can rewrite the entire chain from scratch. Anchoring a
digest on-chain removes that last trust assumption, because the anchor lives
outside our control - see `blockchain.py`.

Attribution
-----------
`actor` only means something once requests are authenticated. Until then entries
are recorded as ``unauthenticated`` and the API says so, rather than inventing an
identity. A forged audit trail would be worse than an obviously incomplete one.
"""

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy import text

from models import CustodyEntry

GENESIS_HASH = "0" * 64

# --- event vocabulary -------------------------------------------------------
ASSET_REGISTERED = "ASSET_REGISTERED"
ANCHOR_REQUESTED = "ANCHOR_REQUESTED"
ANCHOR_CONFIRMED = "ANCHOR_CONFIRMED"
ANCHOR_FAILED = "ANCHOR_FAILED"
CHECK_PERFORMED = "CHECK_PERFORMED"
AMBIGUITY_RESOLVED = "AMBIGUITY_RESOLVED"
CASE_ATTACHED = "CASE_ATTACHED"
REPORT_EXPORTED = "REPORT_EXPORTED"
NOTE_ADDED = "NOTE_ADDED"

ANCHOR_EVENTS = frozenset({ANCHOR_REQUESTED, ANCHOR_CONFIRMED, ANCHOR_FAILED})

SUBJECT_ASSET = "asset"
SUBJECT_EVIDENCE = "evidence"

ANONYMOUS = "unauthenticated"

# Human-readable labels, kept beside the vocabulary so the API and the UI cannot
# drift apart.
EVENT_LABELS = {
    ASSET_REGISTERED: "Original registered",
    ANCHOR_REQUESTED: "Anchoring requested on-chain",
    ANCHOR_CONFIRMED: "Anchored on-chain",
    ANCHOR_FAILED: "Anchoring failed",
    CHECK_PERFORMED: "Comparison performed",
    AMBIGUITY_RESOLVED: "Matched original chosen by user",
    CASE_ATTACHED: "Attached to a case",
    REPORT_EXPORTED: "Report exported",
    NOTE_ADDED: "Note added",
}


def canonical_json(value) -> str:
    """Deterministic JSON: key order and whitespace must not change the hash."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str
    )


def _iso(moment: datetime) -> str:
    """Normalise to naive UTC so the hash survives a database round trip."""
    if moment.tzinfo is not None:
        moment = moment.astimezone(timezone.utc).replace(tzinfo=None)
    return moment.isoformat()


def compute_entry_hash(
    *, prev_hash, subject_type, subject_id, event_type, actor, occurred_at, payload
) -> str:
    """The digest that commits to this entry and, through it, to all before it."""
    material = canonical_json(
        {
            "prev_hash": prev_hash or GENESIS_HASH,
            "subject_type": subject_type,
            "subject_id": str(subject_id),
            "event_type": event_type,
            "actor": actor,
            "occurred_at": _iso(occurred_at),
            "payload": payload or {},
        }
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def append_entry(
    db,
    *,
    subject_type,
    subject_id,
    event_type,
    payload=None,
    actor=ANONYMOUS,
    anchor=None,
):
    """Append one entry to a subject's chain. The caller owns the transaction.

    Takes a transaction-scoped advisory lock per subject. Without it two
    concurrent appends can both read the same predecessor and each claim it,
    forking the chain - and a forked chain no longer proves anything.
    """
    db.execute(
        text("SELECT pg_advisory_xact_lock(hashtextextended(:key, 0))"),
        {"key": f"{subject_type}:{subject_id}"},
    )

    prev_hash = db.execute(
        text(
            """
            SELECT entry_hash FROM custody_entries
            WHERE subject_type = :subject_type AND subject_id = :subject_id
            ORDER BY seq DESC
            LIMIT 1
            """
        ),
        {"subject_type": subject_type, "subject_id": str(subject_id)},
    ).scalar()

    occurred_at = datetime.utcnow()
    body = payload or {}
    entry = CustodyEntry(
        subject_type=subject_type,
        subject_id=subject_id,
        event_type=event_type,
        actor=actor,
        occurred_at=occurred_at,
        payload=body,
        prev_hash=prev_hash,
        entry_hash=compute_entry_hash(
            prev_hash=prev_hash,
            subject_type=subject_type,
            subject_id=subject_id,
            event_type=event_type,
            actor=actor,
            occurred_at=occurred_at,
            payload=body,
        ),
        anchor_status="not_required",
    )
    if anchor:
        entry.anchor_tx_hash = anchor.get("tx_hash")
        entry.anchor_block = anchor.get("block_number")
        entry.anchor_status = anchor.get("status", "not_required")

    db.add(entry)
    db.flush()
    return entry


_SELECT = """
    SELECT seq, id, subject_type, subject_id, event_type, actor, occurred_at,
           payload, prev_hash, entry_hash, anchor_tx_hash, anchor_block, anchor_status
    FROM custody_entries
"""


def list_entries(db, subject_type, subject_id, limit=200):
    rows = db.execute(
        text(
            _SELECT
            + """
            WHERE subject_type = :subject_type AND subject_id = :subject_id
            ORDER BY seq ASC
            LIMIT :limit
            """
        ),
        {
            "subject_type": subject_type,
            "subject_id": str(subject_id),
            "limit": limit,
        },
    ).fetchall()
    return [_row_to_dict(row) for row in rows]


def _row_to_dict(row):
    occurred_at = row.occurred_at
    return {
        "seq": row.seq,
        "id": str(row.id),
        "subject_type": row.subject_type,
        "subject_id": str(row.subject_id),
        "event_type": row.event_type,
        "label": EVENT_LABELS.get(row.event_type, row.event_type),
        "actor": row.actor,
        "occurred_at": occurred_at.isoformat() if occurred_at else None,
        "payload": row.payload or {},
        "prev_hash": row.prev_hash,
        "entry_hash": row.entry_hash,
        "anchor": {
            "tx_hash": row.anchor_tx_hash,
            "block_number": row.anchor_block,
            "status": row.anchor_status,
        },
    }


def verify_chain(entries):
    """Recompute every hash and check each entry links to the one before it.

    Returns a report rather than raising: a broken chain is a finding, not an
    error, and the caller may want to show exactly where it broke.
    """
    previous_hash = None
    problems = []

    for entry in entries:
        if (entry["prev_hash"] or None) != (previous_hash or None):
            problems.append(
                {
                    "seq": entry["seq"],
                    "problem": "broken_link",
                    "expected_prev_hash": previous_hash,
                    "found_prev_hash": entry["prev_hash"],
                }
            )

        recomputed = compute_entry_hash(
            prev_hash=entry["prev_hash"],
            subject_type=entry["subject_type"],
            subject_id=entry["subject_id"],
            event_type=entry["event_type"],
            actor=entry["actor"],
            occurred_at=datetime.fromisoformat(entry["occurred_at"]),
            payload=entry["payload"],
        )
        if recomputed != entry["entry_hash"]:
            problems.append(
                {
                    "seq": entry["seq"],
                    "problem": "hash_mismatch",
                    "recomputed": recomputed,
                    "recorded": entry["entry_hash"],
                }
            )

        previous_hash = entry["entry_hash"]

    return {
        "entries": len(entries),
        "intact": not problems,
        "head_hash": previous_hash,
        "problems": problems,
    }
