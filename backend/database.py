"""Database models and connection for artifact storage."""

import os
import uuid
from datetime import datetime
from enum import Enum as PyEnum
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    String,
    create_engine,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import Session, sessionmaker

# Load environment variables
load_dotenv()

# Database URL
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5433/agentic_poc"
)

# Create engine
engine = create_engine(DATABASE_URL, echo=True)
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()


class ArtifactStatus(str, PyEnum):
    """Status of an artifact."""
    INTERMEDIATE = "intermediate"
    FINAL = "final"


class Artifact(Base):
    """Artifact model for storing generated artifacts."""

    __tablename__ = "artifacts"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    type = Column(String, nullable=False)
    status = Column(Enum(ArtifactStatus), nullable=False, default=ArtifactStatus.FINAL)
    data = Column(JSON, nullable=False)  # JSONB in PostgreSQL
    artifact_metadata = Column(JSON, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


def init_db():
    """Initialize database tables."""
    Base.metadata.create_all(bind=engine)


def get_db() -> Session:
    """Get database session."""
    db = SessionLocal()
    try:
        return db
    finally:
        db.close()


def create_artifact(
    db: Session,
    artifact_type: str,
    data: dict,
    status: ArtifactStatus = ArtifactStatus.FINAL,
    artifact_metadata: Optional[dict] = None
) -> Artifact:
    """Create a new artifact."""
    artifact = Artifact(
        type=artifact_type,
        status=status,
        data=data,
        artifact_metadata=artifact_metadata or {}
    )
    db.add(artifact)
    db.commit()
    db.refresh(artifact)
    return artifact


def get_artifacts(
    db: Session,
    type_filter: Optional[str] = None,
    limit: int = 100
) -> list[Artifact]:
    """Retrieve artifacts with optional type filter."""
    query = db.query(Artifact)

    if type_filter:
        query = query.filter(Artifact.type == type_filter)

    return query.order_by(Artifact.created_at.desc()).limit(limit).all()


def get_artifact_by_id(db: Session, artifact_id: uuid.UUID) -> Optional[Artifact]:
    """Get a single artifact by ID."""
    return db.query(Artifact).filter(Artifact.id == artifact_id).first()
