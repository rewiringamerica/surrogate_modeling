# Scripts

This folder contains scripts for data extraction, model training, feature store creation, and evaluation. Below is the order in which the main scripts should be run:

## Script Execution Order

1. `extract_data_01.py` – Extracts raw data and prepares it for feature engineering.
2. `build_feature_store_02.py` – Processes raw data into structured features used for modeling.
3. `model_training_03.py` – Trains the surrogate model using the prepared feature set.
4. `model_evaluation_04.py` – Evaluates the trained model's performance on test data.

For full training, the latter two scripts are run in a [databricks job](https://4617764665359845.5.gcp.databricks.com/jobs/1097926823028440?o=4617764665359845) on a GPU. Additionally, the scripts are set up to read the most recent versioned tables from the previous step, where tables are suffixed with the version number corresponding to this repo in `pyproject.toml`.

## Additional Scripts
- See the `megastock/` subfolder for the parallel preparation and feature extraction scripts (steps 1 and 2) used for the larger sample set used in inference (see [dohyo repo](https://github.com/rewiringamerica/dohyo)) but not training. 
- The `deprecated/` folder contains older evaluation scripts no longer actively maintained.

For more details on individual scripts, refer to header of each file.