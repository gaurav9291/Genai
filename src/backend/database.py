"""Database connection and session management"""

from sqlalchemy import create_engine, event, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from .models import Base, Document
from .config import settings
import logging
import os
from sqlalchemy import inspect
from typing import Generator

logger = logging.getLogger(__name__)

# Create engine with proper SQLite configuration
engine_kwargs = {
    "echo": False,
    "pool_pre_ping": True
}

if "sqlite" in settings.database_url:
    engine_kwargs.update({
        "connect_args": {"check_same_thread": False},
        "poolclass": StaticPool
    })

engine = create_engine(settings.database_url, **engine_kwargs)

# Enable foreign keys for SQLite
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    if "sqlite" in settings.database_url:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

def create_tables():
    """Create all database tables, with option to drop in dev mode"""
    try:
        # In dev mode, drop tables to ensure schema consistency
        if os.environ.get('DEV_MODE') == 'true':
            logger.info("DEV_MODE enabled: dropping and recreating all tables")
            Base.metadata.drop_all(bind=engine)
        
        # Create all tables
        try:
            Base.metadata.create_all(bind=engine)
        except OperationalError as e:
            if "already exists" in str(e).lower():
                logger.warning("Table creation raced with another process; continuing with existing table.")
            else:
                raise
        
        # Validate schema
        inspector = inspect(engine)
        if "documents" in inspector.get_table_names():
            columns = inspector.get_columns("documents")
            expected_columns = {c.name for c in Document.__table__.columns}
            actual_columns = {c['name'] for c in columns}
            missing_columns = expected_columns - actual_columns
            if missing_columns:
                logger.warning(f"Missing columns in documents table: {missing_columns}")
                # Try to add missing columns
                try:
                    with engine.connect() as conn:
                        for column in missing_columns:
                            column_obj = Document.__table__.c[column]
                            column_type = column_obj.type.compile(engine.dialect)
                            default_value = column_obj.default.arg if column_obj.default else 'NULL'
                            conn.execute(text(f"ALTER TABLE documents ADD COLUMN {column} {column_type} DEFAULT {default_value}"))
                        conn.commit()
                    logger.info("Added missing columns to documents table")
                except Exception as e:
                    logger.error(f"Failed to add missing columns: {e}")
            else:
                logger.info("Database tables created and validated successfully")
        else:
            logger.info("Database tables created successfully")
            
    except Exception as e:
        logger.error(f"Error creating database tables: {e}")
        raise

def get_db() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_db_sync() -> Session:
    """Get database session synchronously"""
    return SessionLocal()

# Health check function
def check_database_health() -> bool:
    """Check if database is accessible and schema is valid"""
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        # Validate documents table schema
        inspector = inspect(engine)
        if "documents" in inspector.get_table_names():
            columns = {c['name'] for c in inspector.get_columns("documents")}
            expected = {c.name for c in Document.__table__.columns}
            if not expected.issubset(columns):
                logger.error(f"Schema validation failed: missing columns {expected - columns}")
                return False
        db.close()
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False
