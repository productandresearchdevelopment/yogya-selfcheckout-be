from fastapi import APIRouter
from app.api.skus import router as skus_router
from app.api.datasets import router as datasets_router
from app.api.recognition import router as recognition_router
from app.api.system import router as system_router
from app.api.training import router as training_router
from app.api.models import router as models_router
from app.api.evaluation import router as evaluation_router

api_router = APIRouter(prefix="/api/v1")
api_router.include_router(skus_router)
api_router.include_router(datasets_router)
api_router.include_router(recognition_router)
api_router.include_router(system_router)
api_router.include_router(training_router)
api_router.include_router(models_router)
api_router.include_router(evaluation_router)
