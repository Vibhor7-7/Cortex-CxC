"""
Database configuration and session management for Cortex.
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from typing import Generator
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database URL from environment or default to SQLite
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./cortex.db")

# Create SQLAlchemy engine
# For SQLite, we need check_same_thread=False to allow FastAPI to use it
engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {},
    echo=False  # Set to True for SQL query logging during development
)

# Create SessionLocal class for database sessions
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create Base class for declarative models
Base = declarative_base()


# Dependency for FastAPI routes
def get_db() -> Generator[Session, None, None]:
    """
    FastAPI dependency that provides a database session.
    Automatically handles session cleanup.
    
    Usage:
        @app.get("/items")
        def get_items(db: Session = Depends(get_db)):
            return db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@contextmanager
def get_db_context():
    """
    Context manager for database sessions.
    Use when you need a database session outside of FastAPI routes.
    
    Usage:
        with get_db_context() as db:
            db.query(Item).all()
    """
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """
    Initialize the database by creating all tables.
    This should be called during application startup.
    """
    from backend import models  # Import models to register them with Base
    Base.metadata.create_all(bind=engine)


def drop_db():
    """
    Drop all database tables.
    WARNING: This will delete all data!
    Use only for development/testing.
    """
    from backend import models  # Import models to register them with Base
    Base.metadata.drop_all(bind=engine)