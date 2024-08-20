# Surrogate Modeling

... is an umbrella term for approximations of building energy models
using Machine Learning (ML) or algorithmic approach, as compared to
(slower) simulations. This repository contains utilities and models
to replicate [ResStock dataset](https://resstock.nrel.gov/)
using Surrogate Modeling.


This code was developed and depends on Databricks. Note that there are two other version of this model that are stored in `deprecated/`: 

1. A simple feed forward network which is in `src/Yearly_model/` _(not maintained)_
2. A CNN model that is built in a ecosystem agnostic way, which constists of all other files _(not maintained)_
   The repo will be reorganized soon to put (2) into a subfolder and make (1) the main set of code in the outer project directory.

More technical documentation will be added in the future, but for now see [Architechture](architecture.md) and [Features & Upgrades](features_upgrades.md).
