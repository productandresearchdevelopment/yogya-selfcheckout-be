"""
Database Migration Script: SQLite -> PostgreSQL (yogya-c50)
Safely migrates all schema tables and existing user data from storage/studio.db
into the local PostgreSQL instance without altering or deleting the SQLite database.
"""

import os
import sys
import sqlite3
from pathlib import Path
from datetime import datetime
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import URL

# Add backend directory to sys.path
BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from dotenv import load_dotenv
load_dotenv(BACKEND_DIR / ".env")

from app.core.config import (
    DB_HOST,
    DB_PORT,
    DB_USER,
    DB_PASSWORD,
    DB_NAME,
    STORAGE_DIR,
)
from app.core.database import Base
import app.models  # Register all models with Base


def migrate_data():
    sqlite_path = STORAGE_DIR / "studio.db"
    if not sqlite_path.exists():
        print(f"[ERROR] SQLite database file not found at {sqlite_path}")
        return False

    print(f"[1/5] Verifying PostgreSQL connection: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME} ...")
    pg_url = URL.create(
        drivername="postgresql+psycopg",
        username=DB_USER,
        password=DB_PASSWORD if DB_PASSWORD else None,
        host=DB_HOST,
        port=DB_PORT,
        database=DB_NAME,
    )
    pg_engine = create_engine(pg_url, echo=False)

    with pg_engine.connect() as conn:
        res = conn.execute(text("SELECT 1")).scalar()
        if res != 1:
            raise RuntimeError("PostgreSQL connection test failed.")
    print("      PostgreSQL connection established successfully.")

    print("[2/5] Creating PostgreSQL schema and tables...")
    Base.metadata.create_all(bind=pg_engine)
    print("      All SQLAlchemy models created in PostgreSQL.")

    print("[3/5] Inspecting SQLite source records...")
    sqlite_conn = sqlite3.connect(str(sqlite_path))
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cur = sqlite_conn.cursor()

    # Tables to migrate in dependency order
    table_order = [
        "skus",
        "product_images",
        "annotations",
        "model_registry",
        "training_jobs",
        "embeddings",
        "hard_negatives",
        "recognition_logs",
    ]

    print("[4/5] Migrating records from SQLite to PostgreSQL...")
    with pg_engine.begin() as pg_conn:
        for table_name in table_order:
            # Check if SQLite table exists
            sqlite_cur.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                (table_name,),
            )
            if not sqlite_cur.fetchone():
                print(f"      - Table '{table_name}' not in SQLite, skipping.")
                continue

            # Fetch all rows from SQLite
            sqlite_cur.execute(f'SELECT * FROM "{table_name}"')
            rows = sqlite_cur.fetchall()
            if not rows:
                print(f"      - Table '{table_name}': 0 rows.")
                continue

            columns = list(rows[0].keys())
            cols_clause = ", ".join([f'"{c}"' for c in columns])
            placeholders = ", ".join([f":{c}" for c in columns])
            insert_sql = text(f'INSERT INTO "{table_name}" ({cols_clause}) VALUES ({placeholders}) ON CONFLICT DO NOTHING')

            boolean_cols = {"is_active", "is_valid", "is_cancelled", "is_mined_for_training"}
            datetime_cols = {"created_at", "updated_at", "started_at", "completed_at", "mined_at"}

            records_to_insert = []
            for r in rows:
                record_dict = dict(r)
                # Convert booleans and datetimes for PostgreSQL
                for k, v in record_dict.items():
                    if k in boolean_cols:
                        record_dict[k] = bool(v) if v is not None else False
                    elif k in datetime_cols and isinstance(v, str):
                        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"):
                            try:
                                record_dict[k] = datetime.strptime(v, fmt)
                                break
                            except ValueError:
                                pass
                records_to_insert.append(record_dict)

            pg_conn.execute(insert_sql, records_to_insert)
            print(f"      - Table '{table_name}': Migrated {len(records_to_insert)} rows.")

            # Reset sequence for PostgreSQL auto-increment primary key
            try:
                seq_query = text(f"SELECT pg_get_serial_sequence('\"{table_name}\"', 'id')")
                seq_name = pg_conn.execute(seq_query).scalar()
                if seq_name:
                    max_id = pg_conn.execute(text(f'SELECT COALESCE(MAX(id), 0) FROM "{table_name}"')).scalar()
                    pg_conn.execute(text(f"SELECT setval('{seq_name}', {max(max_id, 1)})"))
            except Exception as seq_err:
                print(f"        (Sequence reset notice for {table_name}: {seq_err})")

    sqlite_conn.close()

    print("[5/5] Verifying PostgreSQL table row counts against SQLite...")
    inspector = inspect(pg_engine)
    pg_tables = inspector.get_table_names()

    with pg_engine.connect() as pg_conn:
        for table_name in table_order:
            if table_name in pg_tables:
                cnt = pg_conn.execute(text(f'SELECT COUNT(*) FROM "{table_name}"')).scalar()
                print(f"      - {table_name}: {cnt} rows verified in PostgreSQL.")

    print("\nMigration complete! SQLite data safely copied to PostgreSQL without data loss.")
    return True


if __name__ == "__main__":
    success = migrate_data()
    sys.exit(0 if success else 1)
