# \# NYC Airbnb Rental-Price ML Pipeline

# 

# \*\*GitHub repository:\*\* https://github.com/sethgilbert-git/Project-Build-an-ML-Pipeline-Starter

# 

# \*\*W\&B project:\*\* https://wandb.ai/sethc-gilbert-western-governors-university/nyc\_airbnb

# 

# \## Project Overview

# 

# This project estimates historical nightly Airbnb listing prices in New York City using a reusable machine learning pipeline.

# 

# Hydra manages configuration, MLflow executes pipeline components and exports models, and Weights \& Biases tracks experiments, datasets, models, and artifact lineage.

# 

# The pipeline performs the following steps:

# 

# 1\. Fetch a raw data sample.

# 2\. Clean and preprocess the data.

# 3\. Validate the cleaned dataset.

# 4\. Split the data into training, validation, and test sets.

# 5\. Train and compare Random Forest models.

# 6\. Select the model with the lowest validation mean absolute error.

# 7\. Evaluate the selected model on the held-out test set.

# 8\. Run versioned pipeline releases on a new data sample.

# 

# \## Environment Setup

# 

# The project uses Python 3.12 with dependencies specified in `environment.yml` and `requirements.txt`. Linux or Ubuntu through Windows Subsystem for Linux is recommended.

# 

# Create and activate the environment:

# 

# ```bash

# conda env create -f environment.yml

# conda activate nyc\_airbnb\_dev

# ```

# 

# Configure and authenticate W\&B:

# 

# ```bash

# export WANDB\_ENTITY=sethc-gilbert-western-governors-university

# export WANDB\_PROJECT=nyc\_airbnb

# wandb login

# ```

# 

# Enter the W\&B API key only in the terminal login prompt. Credentials are not included in the repository.

# 

# \## Pipeline Configuration

# 

# Pipeline parameters are defined in `config.yaml`, including:

# 

# \- Input sample.

# \- Accepted price range.

# \- Geographic boundaries.

# \- Artifact names and references.

# \- Data-distribution threshold.

# \- Train, validation, and test split settings.

# \- Random seed.

# \- Random Forest hyperparameters.

# 

# Component parameters are passed from configuration rather than hard-coded in the orchestrator.

# 

# The pipeline uses the active environment for its component runs through `env\_manager: local`.

# 

# \## Data Cleaning and Validation

# 

# The cleaning component retains listings priced between $10 and $350 per night and converts review dates to datetime values.

# 

# The geographic fix introduced in version 1.0.1 retains listings within these inclusive boundaries:

# 

# \- Longitude: -74.25 to -73.50.

# \- Latitude: 40.5 to 41.2.

# 

# The data-validation component checks:

# 

# 1\. Expected column names and order.

# 2\. Expected borough names.

# 3\. Valid geographic boundaries.

# 4\. Borough-distribution similarity to a fixed reference dataset.

# 5\. Dataset size greater than 15,000 and less than 1,000,000 rows.

# 6\. Prices within the configured range.

# 

# The first cleaned sample is assigned the `reference` artifact alias. This reference remains fixed when subsequent samples update the `latest` alias.

# 

# \## Artifacts and Lineage

# 

# | Artifact | Producing component | Purpose |

# |---|---|---|

# | `sample.csv` | `download\_file` | Versioned raw sample. |

# | `clean\_sample.csv` | `basic\_cleaning` | Canonical cleaned dataset. |

# | `clean\_data.csv` | `basic\_cleaning` | Additional cleaned artifact matching the alternate rubric name. |

# | `trainval\_data.csv` | `train\_val\_test\_split` | Data used for training and validation. |

# | `test\_data.csv` | `train\_val\_test\_split` | Held-out test data. |

# | `random\_forest\_export` | `train\_random\_forest` | Complete MLflow sklearn model, with artifact type `model\_export`. |

# | `test\_metrics` | `test\_model` | Test metrics and validation/test comparison. |

# 

# The model collection is named `random\_forest\_export`, and its artifact type is `model\_export`.

# 

# The `reference` and `prod` labels are artifact aliases. Pipeline components record input and output dependencies using `run.use\_artifact()` and `run.log\_artifact()`, allowing their relationships to be inspected in W\&B’s Lineage or Graph view.

# 

# \## Exploratory Data Analysis

# 

# The EDA notebook is located at:

# 

# ```text

# src/eda/eda.ipynb

# ```

# 

# It examines dataset structure, missing values, price distributions, borough and room-type summaries, and listing locations.

# 

# Start JupyterLab with:

# 

# ```bash

# jupyter-lab

# ```

# 

# In normal submission mode, the notebook consumes `sample.csv:latest` from W\&B. Local verification mode uses the bundled first sample and explicitly identifies its outputs as local.

# 

# \## Model and Preprocessing

# 

# The inference pipeline contains a preprocessing step followed by a `RandomForestRegressor`.

# 

# Preprocessing includes:

# 

# \- Room-type ordinal encoding.

# \- Most-frequent imputation and one-hot encoding for borough.

# \- Constant imputation for numeric features.

# \- Review-age transformation using a reference date learned from training data.

# \- Missing-title imputation and TF-IDF title features.

# 

# Preprocessing is fitted only on training rows. The review-date transformation uses a fixed training reference so predictions do not depend on the composition of an inference batch.

# 

# The complete preprocessing and regression pipeline is exported in MLflow sklearn format, including the custom feature-transformation code.

# 

# \## Hyperparameter Search and Model Selection

# 

# The Hydra multirun search compares four configurations:

# 

# | Maximum depth | Number of trees |

# |---|---:|

# | 10 | 100 |

# | 10 | 200 |

# | 50 | 100 |

# | 50 | 200 |

# 

# Run the search with:

# 

# ```bash

# mlflow run . --env-manager local \\

# &#x20; -P steps=train\_random\_forest \\

# &#x20; -P hydra\_options="modeling.random\_forest.max\_depth=10,50 modeling.random\_forest.n\_estimators=100,200 -m"

# ```

# 

# Training runs log validation MAE and R² to W\&B. Lower MAE indicates better performance.

# 

# The best model is selected using validation MAE rather than test performance. Its artifact receives the `prod` alias, and its hyperparameters become the defaults in `config.yaml`.

# 

# The selection helper can identify and promote the best model:

# 

# ```bash

# python scripts/select\_best.py

# python scripts/select\_best.py --promote

# ```

# 

# \## Running the Pipeline

# 

# Detailed setup and submission instructions are provided in `SETUP\_AND\_SUBMIT.md`.

# 

# Prepare the initial version and publish the first sample:

# 

# ```bash

# bash scripts/run\_stage.sh first-data

# ```

# 

# Manually add the `reference` alias to the first `clean\_sample.csv` version in W\&B.

# 

# Run data checks, splitting, and the four-model search:

# 

# ```bash

# bash scripts/run\_stage.sh train

# ```

# 

# Promote the validation-selected model:

# 

# ```bash

# python scripts/select\_best.py --promote

# ```

# 

# Evaluate it against the held-out test dataset:

# 

# ```bash

# bash scripts/run\_stage.sh test

# ```

# 

# \## Local Verification Results

# 

# The following measurements were obtained during local verification. Account-specific experiment runs and artifacts can be inspected in the linked W\&B project.

# 

# The first sample contains 20,000 raw rows and 19,001 rows after price filtering.

# 

# | Partition | Rows |

# |---|---:|

# | Training | 12,160 |

# | Validation | 3,040 |

# | Test | 3,801 |

# 

# The best local configuration used a maximum depth of 50 and 200 trees.

# 

# | Metric | Result |

# |---|---:|

# | Validation MAE | $34.2461 |

# | Validation R² | 0.5509 |

# | Test MAE | $33.8417 |

# | Test R² | 0.5623 |

# | Absolute relative MAE gap | 1.18% |

# 

# The validation/test MAE gap is below the configured 15% comparison tolerance. This supports comparable performance on these splits, but does not guarantee future-market accuracy.

# 

# Local verification also confirmed that:

# 

# \- The MLflow model reloads with matching predictions.

# \- The exported model loads in a fresh process outside the repository.

# \- Predictions remain consistent across inference batch compositions.

# \- The original cleaning behavior leaves one out-of-area row in the second sample.

# \- The geographic boundary check detects that row.

# \- After geographic filtering, all six data checks pass and 46,427 rows remain.

# \- Training on the corrected second sample succeeds, with local validation MAE of $32.3596.

# 

# Detailed local results are recorded in `local\_verification/metrics.json`.

# 

# \## Code Tests

# 

# Run the code tests with:

# 

# ```bash

# python -m pytest tests -v

# ```

# 

# Run the more extensive local verification, including model comparison and export checks, with:

# 

# ```bash

# python scripts/local\_verify.py

# ```

# 

# Local verification does not substitute for the required online W\&B runs or published GitHub releases.

# 

# \## Versioned Releases and New-Sample Verification

# 

# The release sequence demonstrates detection and correction of an unexpected data issue.

# 

# \### Version 1.0.0

# 

# The original cleaning behavior filters prices without removing out-of-area coordinates.

# 

# Prepare this version with:

# 

# ```bash

# python scripts/prepare\_release.py 1.0.0

# ```

# 

# After committing, pushing, and publishing release `1.0.0`, run the remote release on the second sample:

# 

# ```bash

# bash scripts/run\_stage.sh second-fail \\

# &#x20; https://github.com/sethgilbert-git/Project-Build-an-ML-Pipeline-Starter.git

# ```

# 

# The expected failure is `test\_proper\_boundaries`, caused by an out-of-area listing in `sample2.csv`.

# 

# \### Version 1.0.1

# 

# The updated cleaning behavior removes records outside the configured NYC boundaries.

# 

# Prepare this version with:

# 

# ```bash

# python scripts/prepare\_release.py 1.0.1

# ```

# 

# After committing, pushing, and publishing release `1.0.1`, run:

# 

# ```bash

# bash scripts/run\_stage.sh second-pass \\

# &#x20; https://github.com/sethgilbert-git/Project-Build-an-ML-Pipeline-Starter.git

# ```

# 

# The corrected release is expected to pass the data checks and train a new model. Actual remote execution status should be verified in W\&B and the published versions in GitHub Releases.

# 

# \## Limitations and Future Improvements

# 

# The model applies to historical NYC listings within the configured $10–$350 price range. It does not cover luxury listings outside that range or properties outside the accepted area.

# 

# Listing prices do not necessarily represent realized booking revenue. Seasonality, fees, amenities, and current demand are not fully represented.

# 

# Future improvements could include temporal validation, richer property features, prediction intervals, drift monitoring, subgroup evaluation, and comparisons with alternative regression models.

# 

# The held-out test set must remain excluded from hyperparameter selection.

