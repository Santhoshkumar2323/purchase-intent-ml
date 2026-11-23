import numpy as np

EPS = 1e-4
QUANTILES = np.linspace(0.1, 0.9, 9)


def _clean(values) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    return values[~np.isnan(values)]


def make_edges(values) -> list[float]:
    values = _clean(values)
    if values.size == 0:
        return []
    return sorted({float(e) for e in np.quantile(values, QUANTILES)})


def proportions(values, edges: list[float]) -> np.ndarray:
    if not edges:
        return np.array([1.0])
    values = _clean(values)
    counts = np.bincount(np.digitize(values, edges, right=True), minlength=len(edges) + 1)
    total = counts.sum()
    return counts / total if total else np.full(len(edges) + 1, 1 / (len(edges) + 1))


def psi(expected, actual) -> float:
    e = np.clip(np.asarray(expected, dtype=float), EPS, None)
    a = np.clip(np.asarray(actual, dtype=float), EPS, None)
    e, a = e / e.sum(), a / a.sum()
    return float(np.sum((a - e) * np.log(a / e)))