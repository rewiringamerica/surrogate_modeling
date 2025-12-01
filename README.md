# Surrogate Modeling

Surrogate modeling is an umbrella term for approximations of building energy models using machine learning (ML) or algorithmic approaches, as compared to (slower) simulations. This repository contains utilities and models to replicate the [ResStock dataset](https://resstock.nrel.gov/) using surrogate modeling techniques.

This code was developed in and depends on Databricks.

## Documentation

The steps run the training pipeline can be found [here](scripts/readme.md), and the details on versioning and model artifacts can be found [here](model_artifacts/readme.md).

More technical documentation is available in the following locations:

- [Architecture Overview](docs/architecture.md)
- [Features & Upgrades](docs/features_upgrades.md)

### Deprecation Notice

There are two deprecated versions of the model stored in `deprecated/` that are no longer maintained.

## Installation

This repository is designed to be run on Databricks and follows the conventions of the [dml-sample-transmission-line](https://github.com/rewiringamerica/dml-sample-transmission-line) repository. Please review its README for details on setup and usage patterns.

We currently run this project on clusters with DB 14.3 LTS runtime (Python 3.10).

### Cluster Setup

To configure the cluster:

1. Upload `install-db-requirements.sh` to Advanced Options > Init Script in your cluster settings.
2. Restart the cluster for changes to take effect.

### Updating Requirements

Whenever you add a requirement to `pyproject.toml`, follow these steps:

1. Run `poetry update`.
2. Generate requirements files with `dml-gen-requirements` as described in the [dml-sample-transmission-line README](https://github.com/rewiringamerica/dml-sample-transmission-line).

### Spell-checker

This repo has [cspell](https://cspell.org/) configured in `cspell.json` for optional (highly recommended) spell-checking. If you're using VSCode, all you need is to install the [cspell extension](https://marketplace.visualstudio.com/items?itemName=streetsidesoftware.code-spell-checker) and spelling issues will be highlighted. If you're using another IDE, [install cspell](https://cspell.org/docs/installation) then run spell-checking in your command line using `cspell .`. To add new word(s) to the dictionary in VSCode, select some text > Right click > Spelling > Add Words to Dictionary. In other IDEs, words may need to be added manually to `/.cspell/sumo_dict.txt`. For more details, see the [cspell docs](https://cspell.org/docs/getting-started).

## Repository Structure

```
├── LICENSE
├── README.md
├── deprecated/                   # Old, unmaintained models
├── docs/                         # Documentation
│   ├── Building_towards_an_MVP.pdf  # Model iteration notes pre-v1.0.0, now this is in release notes
│   ├── architecture.md
│   └── features_upgrades.md
├── images/                       # Architecture diagrams and visuals
├── install-db-requirements.sh    # Cluster init file, used to install `requirements-db-14.3.txt` on databricks
├── model_artifacts/              # Stored model artifacts, including data params and evaluation results
├── notebooks/                    # Jupyter notebooks for analysis
├── poetry.lock                   # Poetry files
├── pyproject.toml                # 
├── scripts                       # Data extraction, training, and evaluation scripts
│   ├── megastock/                # Megastock-specific scripts (See scripts/megastock/README.md)
│   └── deprecated/               # Old scripts, no longer used
├── src/                          # Source code for the surrogate model
│   ├── utils/                    # General utility functions
│   ├── globals.py                # Global variables
│   ├── surrogate_model.py        # Main NN model implementation
│   ├── datagen.py                # Generates training data to feed into NN
│   ├── feature_utils.py          # Feature transformation utilities, used by main training pipeline and megastock
│   ├── versioning.py             # Version control utilities
├── tests/                        # Unit tests
└── requirements-*.txt            # Dependencies
```

## License

This project is licensed under the terms specified in `LICENSE`.