import os
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ValidationError

from src.errors import ConfigError
from src.paths import config_path

ENV_PREFIX = "PIM_"


class DataCfg(BaseModel):
    raw_filename: str = "online_shoppers_intention.csv"
    train_months: list[str]
    seed: int = 42


class FeaturesCfg(BaseModel):
    drop: list[str] = []


class ModelCfg(BaseModel):
    version: str = "v1"
    on_rerun: Literal["overwrite", "new_version", "fail"] = "overwrite"
    type: Literal["xgboost", "logistic_regression"] = "xgboost"
    calibration: Literal["isotonic", "sigmoid", "none"] = "isotonic"
    params: dict[str, Any] = {}
    calibration_months: list[str] = ["Jul", "Aug"]    


class TrainingCfg(BaseModel):
    cv_folds: int = 5
    class_weight: str = "balanced"


class BusinessCfg(BaseModel):
    discount_cost: float
    sale_value: float
    max_lift: float
    lift_ramp_prob: float
    show_cost: float = 0.0    


class DecisionCfg(BaseModel):
    manual_bands: list[dict[str, Any]] | None = None


class ExplainCfg(BaseModel):
    enabled: bool = True
    top_k: int = 5


class DriftCfg(BaseModel):
    psi_warn: float = 0.10
    psi_alert: float = 0.25
    cat_share_alert: float = 0.05
    min_batch_size: int = 200
    ignore: list[str] = []


class ApiCfg(BaseModel):
    log_level: str = "info"


class Settings(BaseModel):
    data: DataCfg
    features: FeaturesCfg
    model: ModelCfg
    training: TrainingCfg
    business: BusinessCfg
    decision: DecisionCfg
    explain: ExplainCfg
    drift: DriftCfg
    api: ApiCfg


def _apply_env(raw: dict) -> dict:
    """PIM_MODEL__VERSION=v2 sets raw['model']['version'] = 'v2'."""
    for key, value in os.environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        *parents, leaf = key[len(ENV_PREFIX):].lower().split("__")
        node = raw
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = yaml.safe_load(value)
    return raw


def load_settings() -> Settings:
    path = config_path()
    try:
        with open(path, encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        if not isinstance(raw, dict):
            raise ConfigError(f"{path} is empty or not a YAML mapping.")
        return Settings(**_apply_env(raw))
    except FileNotFoundError as err:
        raise ConfigError(f"Config file not found: {path}") from err
    except yaml.YAMLError as err:
        raise ConfigError(f"Invalid YAML in config or env override: {err}") from err
    except ValidationError as err:
        raise ConfigError(f"Invalid config values:\n{err}") from err

settings = load_settings()