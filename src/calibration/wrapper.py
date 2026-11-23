import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

EPS = 1e-6


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p)).reshape(-1, 1)


def fit_calibrator(raw, y, method: str):
    if method == "none":
        return None
    if method == "isotonic":
        return IsotonicRegression(out_of_bounds="clip").fit(raw, y)
    return LogisticRegression().fit(_logit(raw), y)  # sigmoid (Platt scaling)


class CalibratedModel:
    def __init__(self, base, calibrator, method: str):
        self.base = base
        self.calibrator = calibrator
        self.method = method

    def predict_proba(self, X) -> np.ndarray:
        raw = self.base.predict_proba(X)[:, 1]
        if self.calibrator is None:
            p = raw
        elif self.method == "isotonic":
            p = self.calibrator.predict(raw)
        else:
            p = self.calibrator.predict_proba(_logit(raw))[:, 1]
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1 - p, p])