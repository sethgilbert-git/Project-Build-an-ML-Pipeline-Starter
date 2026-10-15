#!/usr/bin/env bash
# Install a task-specific Python 3.12 conda environment; never embed API keys.
set -euo pipefail
cd "$(dirname "$0")/.."
if ! command -v conda >/dev/null 2>&1; then
  if [[ -x "$HOME/miniconda3/bin/conda" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
  else
    echo "Conda is not installed. Follow SETUP_AND_SUBMIT.md, step 2."
    exit 2
  fi
fi
source "$(conda info --base)/etc/profile.d/conda.sh"
if conda env list | awk '{print $1}' | grep -qx nyc_airbnb_dev; then
  echo "Reusing nyc_airbnb_dev; updating its packages from requirements.txt."
  conda run -n nyc_airbnb_dev python -m pip install -r requirements.txt
else
  conda env create -f environment.yml
fi
conda run -n nyc_airbnb_dev python -m ipykernel install --user \
  --name nyc_airbnb_dev --display-name "NYC Airbnb (Python 3.12)"
echo "Setup finished. Run: conda activate nyc_airbnb_dev"
echo "Then: wandb login  (paste your key only in your own terminal)"
