from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.core.database import Base, engine, SessionLocal
from app.models.embedding import Embedding
from app.services.gallery import GalleryService
from app.services.model_registry_service import get_active_model

def migrate():
    print("Migrating embeddings table in database...")
    with engine.connect() as conn:
        conn.exec_driver_sql("DROP TABLE IF EXISTS embeddings")
        conn.commit()

    Base.metadata.create_all(bind=engine)
    print("Recreated embeddings table with UniqueConstraint(image_id, model_name).")

    # Reindex with currently active model
    db = SessionLocal()
    active_model = get_active_model(db, "embedding")
    print(f"Active model: {active_model.model_id}")
    gallery_service = GalleryService()
    count = gallery_service.reindex_all_references(db)
    print(f"Indexed {count} reference vectors for {active_model.model_id}.")
    db.close()

if __name__ == "__main__":
    migrate()
