# MegaStock

## Sampled resstock building metadata to MegaStock

Process to generate new files for MegaStock:

1. Generate resstock building samples using the resstock repo. 
   - See the [resstock github repo](https://github.com/NREL/resstock/tree/develop?tab=readme-ov-file), and the [relevant documentation](https://resstock.readthedocs.io/en/latest/basic_tutorial/architecture.html#sampling).
   - Follow their installation instructions -- you'll have to install OpenStudio and the appropriate version of ruby to match what is defined in the resstock repo. They use [rbenv](https://github.com/rbenv/rbenv#readme) to manage ruby versions.
   - generate building metadata csv files using their sampling script
   - Sampled files using v3.3.0 are currently on GCS at `the-cube/data/processed/sampling_resstock/resstock_v3.3.0`. There are files corresponding to sampling with N=10k, 1M, 2M, and 5M. Currently only the 1M file has been completely processed for MegaStock.
2. Run the contents of `data_prep_01` notebook, referencing appropriate file names
   - There are functions applied which:
     - process the resstock v3.3.0 data to match 2022.1 format
     - generate cleaned building metadata tables
3. Run the contents of `feature_extract_02`, referencing appropriate file names
   - There are functions/code which:
     - transform building features and add upgrades and weather city
     - write out building metadata and upgrades to the feature store


To implement the new files in Megastock:
   - Replace the tables queried in lines [419](https://github.com/rewiringamerica/dohyo/blob/4341c4d6fc7ee6914b559b78febb465677421573/src/dohyo.py#L419) and [446](https://github.com/rewiringamerica/dohyo/blob/4341c4d6fc7ee6914b559b78febb465677421573/src/dohyo.py#L446) (and the reference [here](https://github.com/rewiringamerica/dohyo/blob/4341c4d6fc7ee6914b559b78febb465677421573/src/dohyo.py#L375)) with the appropriate tables generated in the above notebooks
  - Then you should be able to run the `dohyo_prototype_demo` notebook with MegaStock now being queried.


## Useful info
- [Reference figma diagram](https://www.figma.com/board/HbgKjS4P6tHGDLmz84fxTK/SuMo%2FDoyho?node-id=9-429&node-type=section&t=UCFHhbgvIyBZKoQM-0)