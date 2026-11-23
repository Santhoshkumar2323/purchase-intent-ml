from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def config_path() -> Path:
    return ROOT / "config.yaml"


def raw_data_path(filename: str) -> Path:
    return ROOT / "data" / "raw" / filename


def artifact_dir(version: str) -> Path:
    return ROOT / "artifacts" / version


def artifact_path(version: str, filename: str) -> Path:
    return artifact_dir(version) / filename


def report_dir(version: str) -> Path:
    return ROOT / "reports" / version


def report_path(version: str, filename: str) -> Path:
    return report_dir(version) / filename


def figure_path(version: str, name: str) -> Path:
    return report_dir(version) / "figures" / f"{name}.png"


def runs_log_path() -> Path:
    return ROOT / "reports" / "runs.csv"