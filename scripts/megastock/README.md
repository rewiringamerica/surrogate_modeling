# MegaStock

## Sampled resstock building metadata to MegaStock

Process to generate new files for MegaStock:

A. Generate resstock building samples using the resstock repo. 
   - See the [resstock github repo](https://github.com/NREL/resstock/tree/develop?tab=readme-ov-file), and the [relevant documentation](https://resstock.readthedocs.io/en/latest/basic_tutorial/architecture.html#sampling).
   - Follow their installation instructions -- you'll have to install OpenStudio and the appropriate version of ruby to match what is defined in the resstock repo. They use [rbenv](https://github.com/rbenv/rbenv#readme) to manage ruby versions.
   - generate building metadata csv files using their sampling script
   - Sampled files using v3.3.0 are currently on GCS at `the-cube/data/processed/sampling_resstock/resstock_v3.3.0`. There are files corresponding to multiple sample sizes including N=10k, 1m, 2m, 5m, 10m, 15m, 20m.

B. Run the [MegaStock Job](https://4617764665359845.5.gcp.databricks.com/jobs/724743198057405?o=4617764665359845) with the job parameter `n_sample_tag` set to the sample size suffix of the CSV from step 1. (e.g, '5m'). Note that all tables will be tagged with the current version number of the repo and the sample tag. This will perform the following: 

1. . Run `data_prep_01` notebook, referencing appropriate file names based on the job parameter. There are functions applied which:
     - process the resstock v3.3.0 data to match the EnergyPlus file format and the aligned 2024.2/2022.1 schema as described in `docs/features_upgrades.md`.
     - generate cleaned building metadata tables
2. Run `feature_extract_02`, referencing appropriate file names based on the job parameter. There are functions/code which:
     - transform building features and add weather city
     - write out building metadata to the feature store
3. Run `write_databricks_to_bigquery_03`, , referencing appropriate file names based on the job parameter. There code will write the following table to BQ:
      - `cube-machine-learning.ds_api_datasets.megastock_combined_baseline_{n_sample_tag}_{version_num}`

## Useful info
- [Reference figma diagram](https://www.figma.com/board/HbgKjS4P6tHGDLmz84fxTK/SuMo%2FDoyho?node-id=9-429&node-type=section&t=UCFHhbgvIyBZKoQM-0)