import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.config import settings
from src.data.clean import clean
from src.data.load import load_raw
from src.data.split import split_by_month, xy
from src.errors import PipelineError
from src.explain.explainer import build_explainer
from src.explain.reasons import original_name
from src.paths import artifact_path
from src.storage import load_model, save_figure

SAMPLE = 1000


def importance_by_feature(explainer, X) -> dict[str, float]:
    values = explainer.shap.shap_values(explainer.prep.transform(X))
    groups: dict[str, list[int]] = {}
    for j, name in enumerate(explainer.names):
        groups.setdefault(original_name(str(name), explainer.categorical), []).append(j)
    scores = {k: float(np.abs(values[:, idx].sum(axis=1)).mean()) for k, idx in groups.items()}
    return dict(sorted(scores.items(), key=lambda kv: kv[1]))


def run() -> dict[str, float]:
    version = settings.model.version
    model = load_model(artifact_path(version, "model.joblib"))
    explainer, reason = build_explainer(model)
    if explainer is None:
        raise PipelineError(reason)
    _, test, _ = split_by_month(clean(load_raw()))
    X, _ = xy(test)
    X = X.sample(min(SAMPLE, len(X)), random_state=settings.data.seed)
    scores = importance_by_feature(explainer, X)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(list(scores), list(scores.values()), color="tab:blue")
    ax.set_xlabel("Mean absolute SHAP value (log-odds, raw model)")
    ax.set_title("What drives predictions overall")
    save_figure(fig, version, "shap_summary")
    return scores


def main() -> None:
    try:
        scores = run()
    except (PipelineError, OSError) as err:
        raise SystemExit(f"Error: {err}")
    print("\nTop features by average impact:")
    for name, value in list(scores.items())[::-1][:6]:
        print(f"  {name:<26}{value:>8.3f}")
    print(f"\nSaved shap_summary.png to reports/{settings.model.version}/figures/")


if __name__ == "__main__":
    main()