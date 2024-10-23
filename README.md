# Surrogate Modeling

... is an umbrella term for approximations of building energy models
using Machine Learning (ML) or algorithmic approach, as compared to
(slower) simulations. This repository contains utilities and models
to replicate [ResStock dataset](https://resstock.nrel.gov/)
using Surrogate Modeling.


This code was developed in and depends on Databricks.

More technical documentation will be added in the future, but for now see [Architechture](docs/architecture.md) and [Features & Upgrades](docs/features_upgrades.md).

Note that there are two other version of this model that are stored in `deprecated/` that are not being maintained. 

## Install

This repo is designed to be run on Databricks. But it is also designed to use
shared library code from the RA private repo to avoid duplication of utility
code from one project to another. It follows the conventions described in
[dml-sample-transmission-line](https://github.com/rewiringamerica/dml-sample-transmission-line)
so you should read an understand the conventions and usage patterns described in
the `README.md` found there. We currently run this project on clusters with DB
14.3 LTS runtime (Python 3.10, R 4.2).

You should add `install-db-requirements.sh` as your cluster init script by uploading it in Advanced Options > Init Script in your cluster menu. 

### Updating Requirements

Whenever you add a requirement to `pyproject.toml` you need to

1. `poetry update` as normal
2. generate requirements files with `dml-gen-requirements` as described in the
   [dml-sample-transmission-line](https://github.com/rewiringamerica/dml-sample-transmission-line)
   `README.md`
