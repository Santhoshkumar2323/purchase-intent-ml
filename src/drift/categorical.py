import pandas as pd


def shares(values) -> dict[str, float]:
    counts = pd.Series(values).astype(str).value_counts(normalize=True)
    return {str(k): float(v) for k, v in counts.items()}


def max_shift(expected: dict, actual: dict) -> tuple[float, str | None]:
    categories = set(expected) | set(actual)
    if not categories:
        return 0.0, None
    shifts = {c: abs(actual.get(c, 0.0) - expected.get(c, 0.0)) for c in categories}
    worst = max(shifts, key=shifts.get)
    return round(shifts[worst], 4), worst