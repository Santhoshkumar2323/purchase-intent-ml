from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from src.config import settings
from src.errors import ConfigError
from src.features.pipeline import build_preprocessor
from src.logs import get_logger

log = get_logger(__name__)


def _balanced() -> bool:
    return str(settings.training.class_weight).lower() == "balanced"


def _logistic() -> Pipeline:
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced" if _balanced() else None,
        random_state=settings.data.seed,
    )
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def _xgboost(y) -> Pipeline:
    from xgboost import XGBClassifier 

    params = dict(settings.model.params)
    if _balanced():
        positives = float(y.sum())
        params.setdefault("scale_pos_weight", (len(y) - positives) / max(positives, 1.0))
    params.setdefault("random_state", settings.data.seed)
    try:
        clf = XGBClassifier(eval_metric="logloss", n_jobs=1, **params)
    except TypeError as err:
        raise ConfigError(f"model.params has an invalid XGBoost option: {err}") from err
    return Pipeline([("prep", build_preprocessor()), ("clf", clf)])


def make_estimator(y, kind: str | None = None) -> Pipeline:
    return _xgboost(y) if (kind or settings.model.type) == "xgboost" else _logistic()


def fit_model(X, y) -> tuple[Pipeline, dict]:
    fallback = None
    if settings.model.type == "xgboost":
        try:
            model = make_estimator(y, "xgboost").fit(X, y)
            return model, {"type": "xgboost", "fallback": None}
        except ConfigError:
            raise  
        except Exception as err:
            fallback = f"{type(err).__name__}: {err}"
            log.warning("XGBoost failed (%s); falling back to logistic regression.", fallback)
    model = make_estimator(y, "logistic_regression").fit(X, y)
    return model, {"type": "logistic_regression", "fallback": fallback}