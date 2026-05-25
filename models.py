import uuid
from datetime import datetime
from sqlalchemy import Column, String, LargeBinary, DateTime
from sqlalchemy.dialects.postgresql import UUID
from database import Base

class OriginalAsset(Base):
    __tablename__ = "original_assets"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    file_name = Column(String(255), nullable=False)
    file_data = Column(LargeBinary, nullable=False) # Maps to BYTEA
    phash = Column(String(16), nullable=False)
    sha256_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)