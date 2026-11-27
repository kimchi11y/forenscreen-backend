import uuid
from datetime import datetime

from sqlalchemy import (
    BigInteger,
    Boolean,
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
    # The account that registered it, as auth.actor_from() spells it
    # ("user:<id>"). Only the owner may see the image or its filename; everyone
    # else's checks can still be matched against it.
    owner = Column(String(128), nullable=False, index=True)
    # Set when this was registered despite a warning that another account had
    # already registered a similar picture: the earlier original. A check that
    # lands exactly on this asset is then not reported as a clean original.
    similar_to = Column(UUID(as_uuid=True), ForeignKey("original_assets.id"), nullable=True)
    # Context recorded at registration, every section labelled by its source
    # (server clock, file EXIF/XMP, browser, opt-in location). See metadata.py.
    registration_metadata = Column(JSONB, nullable=True)
    # The owner's choice to show this original to anyone whose check matches
    # it. Off by default: originals are private unless their owner says so.
    public_original = Column(Boolean, nullable=False, default=False, server_default="false")


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
    # Who ran the check. NULL for an anonymous check from the public page,
    # which is then reachable only through its own unguessable id.
    owner = Column(String(128), nullable=True, index=True)


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


class ShareLink(Base):
    """A link the owner of a check gives to someone (police, a school) so they
    can see the result side by side with the original, without an account.

    Only a SHA-256 of the link's secret is stored, so the database never holds
    a working link. Links expire, and the owner can revoke them at any time.
    """

    __tablename__ = "share_links"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    check_id = Column(UUID(as_uuid=True), ForeignKey("evidence_checks.id"), nullable=False, index=True)
    owner = Column(String(128), nullable=False)
    token_sha256 = Column(String(64), nullable=False, unique=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)
    view_count = Column(Integer, nullable=False, default=0)
    last_viewed_at = Column(DateTime, nullable=True)


class ExtensionToken(Base):
    """A long-lived sign-in for the browser extension, issued from the web app.

    The web app's own tokens last 15 minutes, too short for an extension. This
    one lasts longer but can do one thing only - register originals (see
    auth.EXTENSION_ALLOWED_PATHS) - so a stolen token cannot read anything.
    Only a SHA-256 of it is stored, and it can be revoked.
    """

    __tablename__ = "extension_tokens"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    owner = Column(String(128), nullable=False, index=True)
    token_sha256 = Column(String(64), nullable=False, unique=True)
    label = Column(String(80), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    expires_at = Column(DateTime, nullable=False)
    last_used_at = Column(DateTime, nullable=True)
    revoked_at = Column(DateTime, nullable=True)
