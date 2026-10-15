# Setup and submission — follow in order

This is a separate project from the Census FastAPI assignment. The required submission contains **two public links**: your GitHub repository and your W&B project. Local test logs alone do not fulfill the online rubric.

## 1. Create the GitHub fork and copy this source

Fork https://github.com/udacity/Project-Build-an-ML-Pipeline-Starter into your own GitHub account. Clone it with GitHub Desktop, then use Repository → Show in Explorer. Copy the contents of the `Project-Build-an-ML-Pipeline-Starter` directory from this ZIP into that clone, replacing the starter files. Include `.github`, `.gitignore`, `src`, `components`, `scripts`, `MLproject`, configuration, requirements and documentation. Do not copy the enclosing ZIP folder or any virtual environment.

This download is the finished source, not yet the first release. The following first-data stage switches the cleaning code to 1.0.0 before you commit and publish it.

## 2. Prepare WSL and Python on Windows

Open PowerShell as Administrator and run `wsl --install -d Ubuntu` if WSL is not installed. Restart if prompted. Open Ubuntu and complete its Linux username/password setup. All subsequent commands in this guide are entered in **Ubuntu**, not PowerShell.

If Miniconda is not installed in Ubuntu:

```bash
curl -L https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o /tmp/nyc-miniconda.sh
bash /tmp/nyc-miniconda.sh -b -p "$HOME/miniconda3"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda init bash
```

Navigate to your Windows GitHub clone through `/mnt/c`. If it is in the same location as the earlier project, use:

```bash
cd /mnt/c/Users/Seth/OneDrive/Documents/GitHub/Project-Build-an-ML-Pipeline-Starter
bash scripts/setup_wsl.sh
conda activate nyc_airbnb_dev
python --version
```

Python should report 3.12.x. If your clone is elsewhere, adjust the `cd` path using GitHub Desktop's Show in Explorer location. The project deliberately uses one pinned Python 3.12 environment rather than mixing the older course environment and the newer starter's environment.

## 3. Create and authenticate W&B

Create a personal account at https://wandb.ai. Identify your actual username or team/entity from your profile; it is not necessarily your GitHub username. In Ubuntu run:

```bash
export WANDB_ENTITY=YOUR_ACTUAL_WANDB_USERNAME_OR_TEAM
export WANDB_PROJECT=nyc_airbnb
wandb login
```

Paste the API key from https://wandb.ai/authorize only into the login prompt in your own terminal. Do not put it in source code, GitHub, screenshots or chat. Repeat the export commands when you open a new Ubuntu terminal.

## 4. Publish first-sample data

```bash
bash scripts/run_stage.sh first-data
```

This prepares the original release behavior, uploads sample.csv, and publishes clean_sample.csv plus clean_data.csv. Open `https://wandb.ai/YOUR_ENTITY/nyc_airbnb`. In the project's settings, set its visibility to **Public** and verify an InPrivate window can open it.

## 5. Manually create the fixed reference

In W&B open Artifacts → clean_sample → clean_sample.csv → version with latest. In **Aliases**, add **reference**. Do this on the first cleaned sample. Keep reference attached to this version when sample2 later updates latest. Adding a display tag alone is insufficient; the pipeline resolves clean_sample.csv:reference.

## 6. Run EDA

```bash
jupyter-lab
```

Open `src/eda/eda.ipynb`, select the NYC Airbnb Python kernel if needed, and run all cells. Do not set NYC_EDA_LOCAL=1: submission mode must consume your W&B sample artifact. Save the notebook and shut down Jupyter. Local outputs in the supplied notebook are clearly identified; your rerun creates a real EDA run and artifact dependency.

## 7. Test data, split and run the four-model Hydra search

```bash
bash scripts/run_stage.sh train
```

The data check should show six passing tests. W&B should now contain trainval_data.csv, test_data.csv, and four completed train_random_forest runs with differing depths/tree counts. The training artifacts are named random_forest_export and have type model_export. Their contents include MLmodel and a serialized sklearn pipeline.

The exact sweep command is:

```bash
mlflow run . --env-manager local -P steps=train_random_forest -P hydra_options="modeling.random_forest.max_depth=10,50 modeling.random_forest.n_estimators=100,200 -m"
```

## 8. Select the best validation model and promote it

In W&B sort finished training runs by mae ascending. Choose the lowest validation MAE, not the test MAE. Open that run's output random_forest_export artifact and add **prod** in Aliases.

The helper can perform the same promotion and write the selected settings into config.yaml:

```bash
python scripts/select_best.py --promote
```

It restricts the comparison to finished training runs for the current trainval artifact, avoiding accidental promotion of a model from a different dataset version. Check that config.yaml's max_depth/n_estimators match the chosen run.

## 9. Evaluate the untouched test dataset

```bash
bash scripts/run_stage.sh test
```

Inspect the test_model run's mae, r2, validation_mae, relative_mae_gap and comparable fields. A configured 15% relative absolute MAE difference is used as a diagnostic threshold. If it fails, investigate; do not retune against the test dataset. The local reference execution had validation MAE about $34.25 and test MAE about $33.84.

## 10. Check W&B lineage and add both submission links

In Artifacts open random_forest_export and its **Lineage / Graph view**. Verify it connects to trainval_data.csv, the cleaned sample, and sample.csv through their producing runs. The final test_metrics artifact connects to the production model and test_data.csv. Capture genuine evidence if requested by your course.

Edit the first two link lines in README.md to contain your actual GitHub repository and public W&B project URLs. Remove the pending status statement only after all remote requirements are finished. You can retain local verification details as supplemental evidence.

## 11. Commit the original version and publish release 1.0.0

In GitHub Desktop commit the completed files, executed notebook, selected config and README to main, then Push origin. In your GitHub repository select Releases → Create a new release, create tag **1.0.0** targeting main, title it **1.0.0**, and publish it.

Confirm `VERSION` says 1.0.0 and the cleaning function contains no NYC_FILTER block in that tagged commit. The point of this release is to show the second sample's problem was detected before the fix.

## 12. Run the released original pipeline on sample2 — expect failure

In Ubuntu:

```bash
bash scripts/run_stage.sh second-fail https://github.com/YOUR_USERNAME/Project-Build-an-ML-Pipeline-Starter.git
```

This runs the **remote 1.0.0 release**, not your local working tree, with sample2.csv. The expected failure is `test_proper_boundaries` because exactly one price-cleaned row is outside NYC. Keep the terminal output and failed W&B data_tests run. If a different error occurs, fix the setup before treating it as the expected failure.

## 13. Add the fix and publish release 1.0.1

```bash
python scripts/prepare_release.py 1.0.1
python -m pytest tests -v
```

Now the cleaning code includes the geographic filter and VERSION is 1.0.1. Use GitHub Desktop to commit **Filter out-of-area listings after sample2 boundary failure**, then push. Publish another GitHub release using tag **1.0.1** on this new commit. Do not move or overwrite the old 1.0.0 tag.

## 14. Run the released fix on sample2 — expect success

```bash
bash scripts/run_stage.sh second-pass https://github.com/YOUR_USERNAME/Project-Build-an-ML-Pipeline-Starter.git
```

All six data checks should now pass and a new trained model should appear in W&B. The sample2 model remains a candidate: it is not automatically assigned prod. The old production model's test evidence remains associated with sample1. If promoting a sample2 model later, select within its own cohort and evaluate with its matching held-out split.

## 15. Final submission check

Confirm both GitHub releases exist, the first sample2 run failed for the boundary check, the fixed remote release completed training, and W&B is publicly accessible. Update README with an accurate completion status and both links, commit and push. If these final documentation changes are important for the tagged deliverable, publish an additional patch release, retaining both earlier tags.

Submit your **GitHub repository URL** and include **both the GitHub URL and public W&B project URL** in Submission Details. The prepared ZIP is not a substitute for the live W&B artifacts or GitHub releases.
