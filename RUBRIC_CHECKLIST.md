# Rubric mapping

| Requirement | Implementation | Evidence still to create in your account |
|---|---|---|
| Public nyc_airbnb project, two public links | Project configured in config.yaml; README link slots | W&B public visibility and real links |
| Raw sample.csv | components/get_data/run.py | First-data online run |
| EDA | src/eda/eda.ipynb, locally executed outputs | Rerun using the W&B artifact |
| Typed, documented cleaning parameters | src/basic_cleaning/run.py, MLproject | Online basic_cleaning run |
| Config-driven orchestrator | main.py and config.yaml | Successful component runs |
| Cleaned CSV artifact | clean_sample.csv and clean_data.csv | Online logged artifacts |
| Fixed reference | clean_sample.csv:reference used by data checks | Manually add alias on first sample |
| Row-count and price-range tests | src/data_check/test_data.py | All six W&B data tests pass |
| Trainval/test split | components/train_val_test_split | trainval_data.csv and test_data.csv artifacts |
| Imputer and OneHotEncoder | non_ordinal_categorical_preproc | Training code and runs |
| Fitted preprocessing/RF pipeline | get_inference_pipeline and go | Online training runs |
| MLflow model export | save_pipeline; custom transforms packaged | random_forest_export artifact, type model_export |
| MAE/R2 logging | train_random_forest/run.py | Run summary fields |
| More than two Hydra jobs | 2×2 depth/tree sweep | Four completed W&B runs |
| Best-model prod alias | select_best.py plus manual W&B instructions | prod on validation-best model |
| Test/validation comparison | main.test_regression_model and component | Online test_model metrics and test_metrics artifact |
| Graph view | Explicit use_artifact/log_artifact dependencies | Inspect W&B lineage |
| Release original pipeline | prepare_release.py 1.0.0 | Public GitHub release/tag |
| Original release fails on sample2 | second-fail uses remote tag 1.0.0 | Failed boundary test in online run |
| New geographic cleaning | prepare_release.py 1.0.1 | Committed fix in separate tagged commit |
| Fixed release trains sample2 | second-pass uses remote tag 1.0.1 | Successful W&B training and new model |

The code and recorded local verification are complete. Account-specific evidence is pending until the online stages and published releases are actually performed.
