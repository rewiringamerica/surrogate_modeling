# Neural Net Architecture, Parameters, & Prediction Post-Processing

## Architecture

The architecture consist of three components: a **weather time-series convolutional neural network (CNN)**, a **building metadata feed forward network (FNN)**, and a **combine FNN**. The model is trained on the multivariate regression task of predicting total household energy consumption by fuel type, specifically outputting the length 4 vector: [`electricity`,`natural_gas` ,`fuel_oil`,`propane`] . The input features are described in detail [here](https://www.notion.so/Features-Upgrades-c8239f52a100427fbf445878663d7135?pvs=21).

![Diagram](architecture.svg)

The model was inspired by [this multi-modal model](https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/), which takes in numeric, categorical and image data and uses different branches to handle the mixed data. Note that the architecture has been tweaked manually, but no formal experiments have been performed to optimize architecture. This is planned for future work.

**Weather branch (CNN):**

In this case, instead of running 2D convolutions the (first two) spatial dimensions of 3D images (width x height x RGB channel), we run 1D convolutions over the (first) time-dimension of 2D time-series (hour of the year x weather feature). The CNN compresses the entire 8670 x 4 input matrix into a 1 x 8 “weather embedding” which contains the most important aspects of the weather data for the downstream prediction task. Since this is by far the most compute-intensive portion of the model, one option for expediting training is to train the whole model on a small subset of the data, freeze the embeddings, and then swap in this pre-trained embedding model for the CNN branch. This is planned for future work.

**Building metadata branch (FNN)**

Categorical features are encoded as one-hot vectors, combined with numerical features, and passed through 4 dense layers of increasingly smaller size, compressing the entired building metadata feature space into a 8x1 vector.

**Combine trunk (FNN)**

The weather embedding is concatenated with the building metadata embedding, and then passed through 5 dense layers, expanding then contracting down the the 4x1 target of total consumption by fuel.

## Parameters

**Batch size**: 256

**Epochs**: 100, with patience set to 10 tracking validation loss.

**Initialization**: All trainable layers use [He Initialization](https://paperswithcode.com/method/he-initialization).

**Activation**: All trainable layers use a leaky ReLU activation function.

**Optimizer**: Adam

**Loss**: Masked mean absolute error, where `y_pred` is set to 0 if `y_true` is 0.

All other parameters use keras defaults.

## Prediction Post-Processing

The final layer is a leaky ReLU meaning that the model can predict negative values, but predictions are clipped at 0 in post-processing. Values are also set to 0 in post processing if the given fuel target is not used by any appliance in the modeled building.
