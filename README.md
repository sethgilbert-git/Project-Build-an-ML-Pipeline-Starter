# NYC Airbnb Rental-Price ML Pipeline

GitHub repository: https://github.com/sethgilbert-git/Project-Build-an-ML-Pipeline-Starter

W&B project: https://wandb.ai/sethc-gilbert-western-governors-university/nyc_airbnb

## Project Overview

This project estimates historical nightly Airbnb listing prices in New York City using a reusable machine learning pipeline.

Hydra manages configuration, MLflow executes pipeline components and exports models, and Weights & Biases records experiments and artifact lineage.

The pipeline downloads data, cleans it, validates it, creates training and test partitions, compares Random Forest models, and evaluates the validation-selected model on held-out test data.

## Environment

The project uses Python 3.12. Dependencies are specified in environment.yml and requirements.txt.

Linux or Ubuntu through Windows Subsystem for Linux is recommended. The pipeline components use the active environment through the configured env_manager setting.

Detailed setup and execution instructions are available in SETUP_AND_SUBMIT.md.

## Configuration

Pipeline parameters are stored in config.yaml, including:

- Input sample and artifact references.
- Minimum and maximum prices.
- Geographic boundaries.
- Data-distribution thresholds.
- Train, validation, and test split settings.
- Random seed.
- Random Forest hyperparameters.

The main.py orchestrator passes component parameters from this configuration.

## Data Cleaning

The cleaning component retains listings priced between $10 and $350 per night and normalizes review dates.

Version 1.0.1 adds geographic filtering using these inclusive boundaries:

- Longitude: -74.25 to -73.50.
- Latitude: 40.5 to 41.2.

The first sample contains 20,000 raw records and 19,001 records after price filtering.

The second sample contains one out-of-area record after price filtering. Geographic filtering removes that record, leaving 46,427 cleaned records.

## Exploratory Data Analysis

The EDA notebook is located at src/eda/eda.ipynb.

It examines dataset structure, missing values, price distributions, prices by borough and room type, and listing locations.

The W&B EDA run records the first sample's 20,000 rows, 16 columns, price distribution, and listing-location visualization.

## Data Validation

The data-check component implements six tests:

1. Expected column names and order.
2. Expected borough names.
3. Valid geographic boundaries.
4. Borough-distribution similarity to the reference dataset.
5. Dataset size greater than 15,000 and less than 1,000,000 rows.
6. Prices within the configured range.

The first cleaned sample has the reference artifact alias. This remains fixed while later samples update the latest alias.

## Artifacts and Lineage

| Artifact | Producing component | Purpose |
|---|---|---|
| sample.csv | download_file | Versioned raw sample |
| clean_sample.csv | basic_cleaning | Canonical cleaned dataset |
| clean_data.csv | basic_cleaning | Additional cleaned artifact matching the alternate rubric name |
| trainval_data.csv | train_val_test_split | Data used for training and validation |
| test_data.csv | train_val_test_split | Held-out test data |
| random_forest_export | train_random_forest | Complete MLflow sklearn inference pipeline |
| test_metrics | test_model | Held-out evaluation and validation/test comparison |

The random_forest_export collection contains artifacts of type model_export.

Components record input dependencies with run.use_artifact() and output artifacts with run.log_artifact(). These relationships can be inspected in W&B's Lineage or Graph view.

## Dataset Splitting

The first cleaned sample is split into trainval and test partitions. The trainval partition is subsequently divided into training and validation data.

| Partition | Records |
|---|---:|
| Training | 12,160 |
| Validation | 3,040 |
| Test | 3,801 |

The splits use a fixed random seed and borough stratification. The test dataset is excluded from hyperparameter selection.

## Model and Preprocessing

The inference pipeline consists of a preprocessing step followed by a RandomForestRegressor.

Preprocessing includes:

- Ordinal encoding of room type.
- Most-frequent imputation followed by one-hot encoding of borough.
- Constant imputation of numeric features.
- Review-age transformation using a reference date learned from training data.
- Missing-title imputation and TF-IDF title features.

Preprocessing is fitted only on training records. The review-date reference remains fixed during inference so a record's prediction does not depend on other records in its inference batch.

The complete fitted pipeline is exported using MLflow's sklearn model format, including the custom feature-transformation code.

## Hyperparameter Search

The Hydra multirun search produced four completed training runs.

| Maximum depth | Number of trees | Validation MAE |
|---|---:|---:|
| 10 | 100 | $34.5094 |
| 10 | 200 | $34.4025 |
| 50 | 100 | $34.3368 |
| 50 | 200 | $34.2461 |

The model with the lowest validation MAE used a maximum depth of 50 and 200 trees.

Its artifact, random_forest_export:v3, has the prod alias. The selected hyperparameters are recorded in config.yaml.

The selection helper compares validation MAE among completed training runs. Test performance is not used to choose hyperparameters.

## Held-Out Test Evaluation

The selected first-sample model was evaluated on 3,801 held-out records.

| Metric | Result |
|---|---:|
| Validation MAE | $34.2461 |
| Validation R² | 0.5509 |
| Test MAE | $33.8417 |
| Test R² | 0.5623 |
| Absolute relative MAE gap | 1.18% |

The test evaluation recorded comparable=true because the relative MAE gap was below the configured 15% tolerance.

These results support comparable validation and test performance on these splits. They do not guarantee future-market accuracy or prove the absence of overfitting.

## Versioned Pipeline Releases

Two pipeline releases are published in GitHub:

- 1.0.0: Initial pipeline with price cleaning and geographic data validation.
- 1.0.1: Updated pipeline that removes listings outside the configured NYC boundaries.

The geographic filter is absent from the original release and added in the second release.

The release-verification procedure runs version 1.0.0 on sample2.csv to expose the out-of-area record through test_proper_boundaries, then runs version 1.0.1 to verify the correction.

W&B records a completed training run on the corrected second sample. That run used 29,712 training records and 7,429 validation records, with validation MAE of $32.3596 and validation R² of 0.5796.

The corrected second-sample model is a new candidate model. The prod alias remains on the validation-selected first-sample model associated with the reported held-out test evaluation.

## Local Code Verification

Code tests are located in the tests directory.

The local verification script also checks data validation, four model configurations, held-out evaluation, model serialization, and inference consistency.

Local verification confirmed that the exported MLflow model reloads with matching predictions and can load in a fresh process outside the repository.

Detailed local results are available in local_verification/metrics.json. Online experiment records are available in the linked W&B project.

## Limitations and Future Improvements

The model applies to historical NYC listings within the configured $10–$350 price range and geographic boundaries.

Listing prices do not necessarily represent realized booking revenue. Seasonality, fees, amenities, and current demand are not fully represented.

Future improvements could include temporal validation, richer property features, prediction intervals, drift monitoring, subgroup evaluation, and comparisons with alternative regression models.

The held-out test dataset must remain excluded from hyperparameter selection.

## Submission Links

GitHub: https://github.com/sethgilbert-git/Project-Build-an-ML-Pipeline-Starter

Weights & Biases: https://wandb.ai/sethc-gilbert-western-governors-university/nyc_airbnb
