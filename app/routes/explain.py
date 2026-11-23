import pandas as pd
from fastapi import APIRouter, Depends, HTTPException

from app.deps import Artifacts, get_artifacts
from app.schemas import ExplainOut, SessionIn
from src.decision.bands import action_for
from src.logs import get_logger

router = APIRouter()
log = get_logger(__name__)


@router.post("/explain", response_model=ExplainOut)
def explain(session: SessionIn, art: Artifacts = Depends(get_artifacts)) -> ExplainOut:
    if art.explainer is None:
        raise HTTPException(status_code=503, detail=art.explainer_error or "Explainer unavailable.")
    try:
        frame = pd.DataFrame([session.to_row()])
        probability = float(art.model.predict_proba(frame)[0, 1])
        reasons = art.explainer.explain(frame)
    except Exception as err:
        log.error("Explanation failed: %s", err)
        raise HTTPException(status_code=500, detail="Explanation failed.") from err
    return ExplainOut(
        probability=round(probability, 4),
        action=action_for(probability, art.bands),
        model_version=art.version,
        reasons=reasons,
    )