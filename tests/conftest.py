import pytest
from app.core.database import Base, engine

@pytest.fixture(autouse=True, scope="session")
def initialize_database():
    Base.metadata.create_all(bind=engine)
    yield
