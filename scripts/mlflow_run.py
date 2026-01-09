"""
MLflow Configuration.

This file is automatically imported by MLflow to set the correct tracking URI.
To start MLflow UI with the correct configuration, run:
    python -m mlflow_config
"""

import os
import sys
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).parent

# MLflow tracking configuration
MLFLOW_TRACKING_URI = f"file:{PROJECT_ROOT}/data/mlflow/tracking"
MLFLOW_ARTIFACT_LOCATION = f"{PROJECT_ROOT}/data/mlflow/artifacts"

# Set environment variable for MLflow
os.environ["MLFLOW_TRACKING_URI"] = MLFLOW_TRACKING_URI

# Default experiment name
DEFAULT_EXPERIMENT = "lstm-multi-ticker"


def start_ui(port: int = 5001, host: str = "127.0.0.1"):
    """Start MLflow UI with correct configuration."""
    import subprocess
    
    print(f"🚀 Starting MLflow UI...")
    print(f"📂 Tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"🌐 Server: http://{host}:{port}")
    print(f"📊 Default Experiment: {DEFAULT_EXPERIMENT}")
    print("\nPress CTRL+C to stop\n")
    
    try:
        subprocess.run([
            "mlflow", "ui",
            "--backend-store-uri", MLFLOW_TRACKING_URI,
            "--host", host,
            "--port", str(port)
        ])
    except KeyboardInterrupt:
        print("\n\n✅ MLflow UI stopped")
        sys.exit(0)


if __name__ == "__main__":
    # When run directly, start MLflow UI
    import argparse
    
    parser = argparse.ArgumentParser(description="Start MLflow UI with project configuration")
    parser.add_argument("--port", type=int, default=5001, help="Port for MLflow UI")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for MLflow UI")
    
    args = parser.parse_args()
    start_ui(port=args.port, host=args.host)
