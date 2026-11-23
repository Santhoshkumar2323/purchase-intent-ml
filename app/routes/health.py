from fastapi import APIRouter, Request

from app.schemas import HealthOut
from src.config import settings

router = APIRouter()


@router.get("/health", response_model=HealthOut)
def health(request: Request) -> HealthOut:
    art = getattr(request.app.state, "artifacts", None)
    if art is None:
        return HealthOut(
            status="unavailable",
            model_version=settings.model.version,
            detail=getattr(request.app.state, "load_error", "Model not loaded."),
        )
    return HealthOut(
        status="ok",
        model_version=art.version,
        trained_at=art.metadata.get("trained_at"),
    )