# Yogya Self-Checkout Backend (`yogya-selfcheckout-be`)

Backend service for real-time grocery product localization and visual similarity recognition using Faster R-CNN and Meta DINO embeddings.

## Tech Stack
- **Framework:** FastAPI (Python 3.10+)
- **Database:** PostgreSQL (SQLAlchemy ORM)
- **Object Detection:** Faster R-CNN MobileNetV3-Large-FPN (Pretrained COCO)
- **Visual Embedding:** Meta DINO Vision Transformer (ViT-S/14, 384-dim, L2-normalized)
- **Similarity Metric:** Cosine similarity against enrolled SKU reference galleries

## Setup & Running Locally

1. **Create Virtual Environment & Install Dependencies:**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure Environment:**
   Copy `.env.example` to `.env` and fill in your PostgreSQL credentials:
   ```dotenv
   DB_HOST=localhost
   DB_PORT=5433
   DB_USER=postgres
   DB_PASSWORD=your_password
   DB_NAME=yogya-c50
   ```

3. **Run Dev Server:**
   ```bash
   python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
   ```

4. **API Documentation:**
   Interactive Swagger docs available at `http://127.0.0.1:8000/docs`.

## Running Tests
```bash
pytest tests/
```
