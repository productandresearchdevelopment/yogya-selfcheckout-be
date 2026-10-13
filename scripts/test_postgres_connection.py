"""
PostgreSQL Connectivity & Metadata Persistence Verification Test
Tests localhost:5433 / yogya-c50 via SQLAlchemy engine and ORM Session.
"""

import sys
from pathlib import Path
from sqlalchemy import text

BACKEND_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BACKEND_DIR))

from app.core.config import DB_HOST, DB_PORT, DB_USER, DB_NAME, DATABASE_URL
from app.core.database import engine, SessionLocal
from app.models.sku import SKU


def test_postgresql():
    print("=" * 60)
    print(" Testing PostgreSQL Database Connection")
    print(f" Target: {DB_USER}@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    print(f" Dialect: {engine.dialect.name} (driver: {engine.dialect.driver})")
    print("=" * 60)

    # 1. Test raw SELECT 1 connectivity
    print("[1/4] Executing SELECT 1 health probe...")
    with engine.connect() as conn:
        res = conn.execute(text("SELECT 1")).scalar()
        assert res == 1, f"Expected 1, got {res}"
        version = conn.execute(text("SELECT version();")).scalar()
        print(f"      Connected! PostgreSQL Version:\n      {version}")

    # 2. Verify schema and table listing
    print("[2/4] Verifying tables in public schema...")
    with engine.connect() as conn:
        tables = conn.execute(text("""
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public' 
            ORDER BY table_name;
        """)).fetchall()
        table_names = [t[0] for t in tables]
        print(f"      Found {len(table_names)} tables in PostgreSQL:")
        for t in table_names:
            cnt = conn.execute(text(f'SELECT COUNT(*) FROM "{t}"')).scalar()
            print(f"        - {t}: {cnt} rows")

    # 3. Test Metadata Persistence (CRUD on SKU table)
    print("[3/4] Testing SKU metadata persistence transaction...")
    db = SessionLocal()
    test_sku_code = "SKU-POSTGRES-VERIFY-001"
    try:
        # Check if already exists from prior run and clean up
        existing = db.query(SKU).filter(SKU.sku_code == test_sku_code).first()
        if existing:
            db.delete(existing)
            db.commit()

        # Insert test SKU
        test_sku = SKU(
            sku_code=test_sku_code,
            name="PostgreSQL Verification Test Product",
            category="Test / Verification",
            description="Created by test_postgres_connection.py to verify persistence in yogya-c50",
            is_active=True,
        )
        db.add(test_sku)
        db.commit()
        db.refresh(test_sku)
        persisted_id = test_sku.id
        print(f"      Successfully inserted test SKU '{test_sku_code}' with ID: {persisted_id}")

        # Query back directly from PostgreSQL
        queried = db.query(SKU).filter(SKU.id == persisted_id).first()
        assert queried is not None
        assert queried.sku_code == test_sku_code
        print(f"      Verified persistence query: {queried.sku_code} ({queried.name})")

        # Clean up test SKU
        db.delete(queried)
        db.commit()
        print("      Cleaned up verification test SKU cleanly.")

    finally:
        db.close()

    # 4. Confirm strictly PostgreSQL (no SQLite fallback)
    print("[4/4] Verifying active dialect...")
    assert engine.dialect.name == "postgresql", f"Dialect is {engine.dialect.name}, NOT postgresql!"
    print("      Strictly executing against PostgreSQL (no SQLite fallback).")

    print("\nALL POSTGRESQL VERIFICATION CHECKS PASSED!\n")
    return True


if __name__ == "__main__":
    success = test_postgresql()
    sys.exit(0 if success else 1)
