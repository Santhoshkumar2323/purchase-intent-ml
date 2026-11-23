import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.deps import Artifacts, get_artifacts
from app.schemas import PredictOut, SessionIn
from src.decision.bands import action_for
from src.logs import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.post("/predict", response_model=PredictOut)
def predict(session: SessionIn, art: Artifacts = Depends(get_artifacts)) -> PredictOut:
    try:
        frame = pd.DataFrame([session.to_row()])
        probability = float(art.model.predict_proba(frame)[0, 1])
    except Exception as err:  
        log.error("Prediction failed: %s", err)
        raise HTTPException(status_code=500, detail="Prediction failed.") from err
    return PredictOut(
        probability=round(probability, 4),
        action=action_for(probability, art.bands),
        model_version=art.version,
    )