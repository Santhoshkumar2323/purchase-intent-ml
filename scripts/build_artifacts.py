import subprocess
import sys

STEPS = [
    "scripts.download_data",
    "src.training.train",
    "src.calibration.calibrate",
    "src.decision.bands",
]


def main() -> None:
    for module in STEPS:
        print(f"\n=== {module} ===", flush=True)
        if subprocess.run([sys.executable, "-m", module]).returncode != 0:
            raise SystemExit(f"Build stopped: step '{module}' failed.")
    print("\nAll artifacts built.")


if __name__ == "__main__":
    main()