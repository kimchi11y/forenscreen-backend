import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    ForeignKey,
    Identity,
    Index,
    Integer,
    Numeric,
    String,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy import LargeBinary

from database import Base


class OriginalAsset(Base):
    __tablename__ = "original_assets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name = Column(String(255), nullable=False)
    file_data = Column(LargeBinary, nullable=False)  # Maps to BYTEA
    phash = Column(String(16), nullable=False)
    sha256_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class EvidenceCheck(Base):
    """One comparison of an uploaded file against the registry.

    These used to be thrown away - `/upload` returned its verdict and stored
    nothing, so a result could not be revisited, could not be exported, and had
    no custody record. Persisting the check is what makes a chain of custody
    possible at all: you cannot log the handling of a record you do not keep.
    """

    __tablename__ = "evidence_checks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    asset_id = Column(UUID(as_uuid=True), ForeignKey("original_assets.id"), nullable=True)
    sha256_hash = Column(String(64), nullable=False, index=True)
    phash = Column(String(16), nullable=False)
    verdict = Column(String(32), nullable=False)
    match_method = Column(String(32), nullable=True)
    orb_inliers = Column(Integer, nullable=True)
    orb_inlier_ratio = Column(Numeric(6, 4), nullable=True)
    phash_distance = Column(Integer, nullable=True)
    saved_filename = Column(String(255), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)


class CustodyEntry(Base):
    """An append-only entry in the chain of custody.

    Each entry commits to the one before it (`entry_hash` covers `prev_hash`),
    so altering or removing any entry invalidates every later hash. That gives
    tamper-evidence inside the database; anchoring a hash on-chain extends the
    same guarantee to cover the operator as well.

    Only the `anchor_*` columns may change after insert - the hashed fields are
    frozen by a trigger (see schema.sql). Re-anchoring is recorded by appending
    a new entry, never by rewriting this one.
    """

    __tablename__ = "custody_entries"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    # Monotonic insertion order. A UUID primary key is random, so it cannot say
    # which entry came first; the chain needs an unambiguous predecessor.
    seq = Column(BigInteger, Identity(always=True), nullable=False, unique=True)
    subject_type = Column(String(16), nullable=False)  # 'asset' | 'evidence'
    subject_id = Column(UUID(as_uuid=True), nullable=False)
    event_type = Column(String(32), nullable=False)
    actor = Column(String(128), nullable=False, default="unauthenticated")
    occurred_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    payload = Column(JSONB, nullable=False, default=dict)

    prev_hash = Column(String(64), nullable=True)  # NULL only for a subject's first entry
    entry_hash = Column(String(64), nullable=False, unique=True)

    anchor_tx_hash = Column(String(66), nullable=True)
    anchor_block = Column(BigInteger, nullable=True)
    anchor_status = Column(String(16), nullable=False, default="not_required")

    __table_args__ = (
        Index("ix_custody_subject", "subject_type", "subject_id", "occurred_at"),
        Index("ix_custody_anchor_status", "anchor_status"),
    )
