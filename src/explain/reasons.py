import numpy as np


def original_name(feature: str, categorical: list[str]) -> str:
    kind, _, name = feature.partition("__")
    if kind == "cat":
        for col in categorical:
            if name.startswith(f"{col}_"):
                return col
    return name


def top_reasons(names, values, categorical: list[str], row: dict, top_k: int) -> list[dict]:
    totals: dict[str, float] = {}
    for feature, value in zip(names, values):
        key = original_name(str(feature), categorical)
        totals[key] = totals.get(key, 0.0) + float(value)
    ranked = sorted(totals.items(), key=lambda kv: abs(kv[1]), reverse=True)[:top_k]
    return [{"feature": key, "value": row.get(key), "effect": round(effect, 4),
             "direction": "up" if effect >= 0 else "down"}
            for key, effect in ranked if not np.isclose(effect, 0.0)]