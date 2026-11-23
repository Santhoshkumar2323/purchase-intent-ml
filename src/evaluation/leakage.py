from src.config import settings
from src.evaluation.metrics import compute_metrics
from src.logs import get_logger
from src.training.models import fit_model

log = get_logger(__name__)

COLUMN = "PageValues"

def pagevalues_test(X_tr, y_tr, X_te, y_te, with_metrics: dict) -> dict:
    if COLUMN in settings.features.drop:
        return {"skipped": f"{COLUMN} is already dropped in config.yaml"}
    original = settings.features.drop
    settings.features.drop = [*original, COLUMN] 
    try:
        model, info = fit_model(X_tr, y_tr)
        without = compute_metrics(y_te, model.predict_proba(X_te)[:, 1])
    except Exception as err:
        log.warning("PageValues test skipped: %s", err)
        return {"skipped": str(err)}
    finally:
        settings.features.drop = original  
    return {"with": with_metrics, "without": without, "model_used": info["type"]}