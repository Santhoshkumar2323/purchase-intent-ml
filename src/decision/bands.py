import numpy as np

from src.config import settings
from src.decision.ev import expected_value
from src.decision.plots import ev_chart
from src.errors import PipelineError
from src.paths import artifact_path
from src.storage import save_json

GRID = np.linspace(0.0, 1.0, 1001)


def derive_bands() -> list[dict]:
    manual = settings.decision.manual_bands
    if manual:  
        return manual
    positive = GRID[expected_value(GRID) > 0]
    if positive.size == 0:
        return [{"max_prob": 1.0, "action": "no_action"}]
    low, high = float(positive.min()), float(positive.max())
    bands = []
    if low > 0:
        bands.append({"max_prob": round(low, 3), "action": "no_action"})
    bands.append({"max_prob": round(high, 3), "action": "show_offer"})
    if high < 1:
        bands.append({"max_prob": 1.0, "action": "no_action"})
    return bands


def action_for(prob: float, bands: list[dict]) -> str:
    for band in bands:
        if prob <= band["max_prob"]:
            return band["action"]
    return bands[-1]["action"]


def main() -> None:
    version = settings.model.version
    try:
        bands = derive_bands()
        save_json({"bands": bands, "business": settings.business.model_dump()},
                  artifact_path(version, "bands.json"))
        ev_chart(bands, version)
    except (PipelineError, OSError) as err:
        raise SystemExit(f"Error: {err}")
    print("\nDerived bands (probability up to max_prob -> action):")
    for band in bands:
        print(f"  <= {band['max_prob']:<6} {band['action']}")
    print(f"\nSaved bands.json to artifacts/{version}/ and ev_vs_probability.png to reports/")


if __name__ == "__main__":
    main()