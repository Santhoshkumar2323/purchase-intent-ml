from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException, Request

from src.config import settings
from src.explain.explainer import build_explainer
from src.paths import artifact_path
from src.storage import load_json, load_model


@dataclass
class Artifacts:
    version: str
    model: Any
    bands: list[dict]
    metadata: dict
    explainer: Any = None
    explainer_error: str | None = None


def load_artifacts() -> Artifacts:
    version = settings.model.version
    model = load_model(artifact_path(version, "model.joblib"))
    explainer, error = build_explainer(model)
    return Artifacts(
        version=version,
        model=model,
        bands=load_json(artifact_path(version, "bands.json"))["bands"],
        metadata=load_json(artifact_path(version, "metadata.json")),
        explainer=explainer,
        explainer_error=error,
    )


def get_artifacts(request: Request) -> Artifacts:
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        reason = getattr(request.app.state, "load_error", "Model not loaded.")
        raise HTTPException(status_code=503, detail=reason)
    return artifacts