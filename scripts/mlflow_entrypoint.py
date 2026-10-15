"""Forward MLflow parameters as correctly separated Hydra CLI arguments."""
import argparse
from pathlib import Path
import shlex
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", required=True)
parser.add_argument("--hydra-options", default="")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
result = subprocess.run(
    [sys.executable, str(root / "main.py"), f"main.steps='{args.steps}'",
     *shlex.split(args.hydra_options)], cwd=root,
)
sys.exit(result.returncode)
