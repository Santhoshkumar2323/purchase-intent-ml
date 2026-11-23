import pandas as pd

from src.config import settings
from src.explain.reasons import top_reasons
from src.features.pipeline import feature_columns
from src.logs import get_logger

log = get_logger(__name__)


class Explainer:
    def __init__(self, prep, shap_explainer):
        self.prep = prep
        self.shap = shap_explainer
        self.names = list(prep.get_feature_names_out())
        self.categorical = feature_columns()[1]

    def explain(self, frame: pd.DataFrame, top_k: int | None = None) -> list[dict]:
        values = self.shap.shap_values(self.prep.transform(frame))
        row = frame.iloc[0].to_dict()
        return top_reasons(self.names, values[0], self.categorical, row,
                           top_k or settings.explain.top_k)


def build_explainer(model) -> tuple[Explainer | None, str | None]:
    if not settings.explain.enabled:
        return None, "Explanations are disabled in config.yaml (explain.enabled)."
    try:
        import shap

        base = getattr(model, "base", model)  
        clf, prep = base.named_steps["clf"], base.named_steps["prep"]
        if type(clf).__name__ != "XGBClassifier":
            return None, f"Explanations need the XGBoost model, found {type(clf).__name__}."
        return Explainer(prep, shap.TreeExplainer(clf)), None
    except Exception as err:
        log.warning("Explainer unavailable: %s", err)
        return None, f"Explainer could not be built: {err}"