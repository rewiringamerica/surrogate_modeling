# Model Artifacts

This directory stores model outputs, including trained model metadata and evaluation metrics. These artifacts are versioned to track improvements over time.

This information is store [in GCS](https://console.cloud.google.com/storage/browser/the-cube/export/surrogate_model/model_artifacts?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&authuser=0&inv=1&invt=Abr7Kg&project=rewiring-america&supportedpurview=project), and a subset of these files are also stored in this folder in git. 

## Structure 

```
model_artifacts/
├── <version>/                     # Versioned folders (e.g., 01_00_00)
│   ├── model.keras                # Only in GCS
│   ├── mappings.json              # Only in GCS
│   ├── features_targets_upgrades.json 
│   ├── metrics_by_upgrade_type.csv
│   ├── test_baseline_features_input.csv # Only in GCS
│   ├── test_upgraded_features.csv # Only in GCS
```

## Versioning
- Each subdirectory corresponds to a specific model version (e.g., `01_00_00`), which correspond to the poetry version of this repo package, as in `pyproject.toml`.
- The versioning follows `major_minor_patch` format.
- Artifacts are stored in Databricks and synchronized here for tracking.

## Files
- `model.keras` – Trained keras surrogate model (only in GCS).
- `mappings.json` – Data dictionaries used by surrogate model and dohyo downstream (only in GCS).
- `features_targets_upgrades.json` – Documents input features, target variables, and upgrades that model version was trained on. This file must be manually created locally when training a new model version. 
- `metrics_by_upgrade_type.csv` – Contains evaluation results on a test set.
- `test_baseline_features_input.csv` - Test data for feature_utils.apply_upgrades() to make sure that apply logic stays in sync between sumo and dohyo (ipnut data)
- `test_upgraded_features.csv` - Test data for feature_utils.apply_upgrades() to make sure that apply logic stays in sync between sumo and dohyo (expected output data)