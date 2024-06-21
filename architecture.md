# Neural Net Architecture, Parameters, & Prediction Post-Processing

## Architecture

The architecture consist of three components: a **weather time-series convolutional neural network (CNN)**, a **building metadata feed forward network (FNN)**, and a **combine FNN**. The model is trained on the multivariate regression task of predicting total household energy consumption by fuel type, specifically outputting the length 4 vector: [`electricity`,`natural_gas` ,`fuel_oil`,`propane`] . The input features are described in detail [here](https://www.notion.so/Features-Upgrades-c8239f52a100427fbf445878663d7135?pvs=21).

![Diagram](architecture.svg)

The model was inspired by [this multi-modal model](https://pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/), which takes in numeric, categorical and image data and uses different branches to handle the mixed data. Note that the architecture has been tweaked manually, but no formal experiments have been performed to optimize architecture. This is planned for future work.

**Weather branch (CNN):** 

In this case, instead of running 2D convolutions the (first two) spatial dimensions of 3D images (width x height x RGB channel), we run 1D convolutions over the (first) time-dimension of 2D time-series (hour of the year x weather feature). The CNN compresses the entire 8670 x 4 input matrix into a “weather embedding” vector of dimension 8 which contains the most important aspects of the weather data for the downstream prediction task. Since this is by far the most compute-intensive portion of the model, one option for expediting training is to train the whole model on a small subset of the data, freeze the embeddings, and then swap in this pre-trained embedding model for the CNN branch. This is planned for future epic. There is a Batchnorm layer after the initialization and each of the convolutional layers before the activation is applied. 

**Building metadata branch (FNN)**

Categorical features are encoded as one-hot vectors, combined with numerical features, and passed through 4 dense layers of increasingly smaller size, compressing the entire building metadata feature space into 16 dimensional “building metadata embedding”. There is a Batchnorm layer after the initialization and each of the dense layers before the activation is applied. 

**Combine trunk (FNN)**

The weather embedding is concatenated with the building metadata embedding to form a 24 length vector, and then passed through 5 dense layers, contracting down to the 4 dim target vector of total consumption by fuel. 

## Parameters

**Batch size**: 256

**Epochs**: 200, with patience set to 15 tracking validation loss. 

**Initialization**: All trainable layers use [He Initialization](https://paperswithcode.com/method/he-initialization).

**Activation**: All trainable layers use a leaky ReLU activation function.

**Optimizer**: Adam

**Loss**: Masked mean absolute error, where elements (sample i, fuel j) are not included in the mean if $y_{\text{true}}^{i,j} = 0$. Note that this this is analagous to setting the values to 0 in post processing at inference time as described below. 

All other parameters use keras defaults. 

## Prediction Post-Processing

The final layer is a leaky ReLU meaning that the model can predict negative values, but predictions are clipped at 0 in post-processing. 

Values are also set to 0 in post processing if the given fuel target is not used by any appliance in the modeled building.