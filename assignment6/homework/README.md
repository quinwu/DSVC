In this assignment you will practice writing backpropagation code, and training
Neural Networks and Convolutional Neural Networks. The goals of this assignment
are as follows:

- understand **Neural Networks** and how they are arranged in layered
  architectures
- understand and be able to implement (vectorized) **backpropagation**
- implement various **update rules** used to optimize Neural Networks
- implement **batch normalization** for training deep networks
- implement **dropout** to regularize networks
- effectively **cross-validate** and find the best hyperparameters for Neural
  Network architecture
- understand the architecture of **Convolutional Neural Networks** and train
  gain experience with training these models on data

### Neural Network modular

The IPython notebook `FullyConnectedNets.ipynb` will introduce you to our modular layer design, and then use those layers to implement fully-connected networks of arbitrary depth. To optimize these models you will implement several popular update rules.

- `FullyConnectedNets.ipynb`
  - `DSVC/layers.py`
  - `DSVC/layer_utils.py`
  - `DSVC/classfifiers/fc_net.py`
  - `DSVC/solver.py`
  - `DSVC/optim.py`

### Batch Normalization

In the IPython notebook `BatchNormalization.ipynb` you will implement batch normalization, and use it to train deep fully-connected networks.

- `BatchNormalization.ipynb`
  - `DSVC/layers.py`
  - `DSVC/classifiers/fc_net.py`

### Dropout

The IPython notebook `Dropout.ipynb` will help you implement Dropout and explore its effects on model generalization.

- `Dropout.ipynb`
  - `DSVC/layers.py`
  - `DSVC/classifiers/fc_net.py`

### Convolution Neural Networks

In the IPython Notebook `ConvolutionalNetworks.ipynb` you will implement several new layers that are commonly used in convolutional networks.

- `ConvolutionalNetworks.ipynb`
  - `DSVC/layers.py`
  - `DSVC/fast_layers.py`
  - `DSVC/layer_utils.py`
  - `DSVC/classifiers/cnn.py`