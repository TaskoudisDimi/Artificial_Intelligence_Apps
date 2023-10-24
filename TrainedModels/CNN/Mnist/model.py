import torch
import torch.nn as nn


class Model(nn.Module):

    def __init__(self):
        super(Model, self).__init__()
        # Defines a ReLU (Rectified Linear Unit) activation function. 
        # A rectified linear unit (ReLU) is an activation function that introduces 
        # the property of non-linearity to a deep learning model and solves the vanishing gradients issue.
        # The inplace=True argument means that the operation is applied in-place, modifying the input tensor directly.
        self.relu = nn.ReLU(inplace=True)
        # Defines a 2x2 max-pooling layer, which reduces the spatial dimensions of the input tensor by a factor of 2
        # It is similar to the convolution layer but instead of taking a dot product between the input and the kernel we take 
        # the max of the region from the input overlapped by the kernel. 
        self.maxpool = nn.MaxPool2d(2, 2)
        # Define two convolutional layers with batch normalization. These layers are used to learn features from the input image.

        # The term convolution refers to the mathematical combination of two functions to produce a third function. 
        # It merges two sets of information. In the case of a CNN, the convolution is performed on the input data with 
        # the use of a filter or kernel (these terms are used interchangeably) to then produce a feature map.
        self.conv1 = nn.Conv2d(1, 32, 5, 1, 2)
        # Batch normalization is a technique to standardize the inputs to a network, 
        # applied to either the activations of a prior layer or inputs directly. Batch normalization accelerates training, in some cases by halving the epochs or better, and provides some regularization, reducing generalization error.
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, 5, 1, 2)
        self.bn2 = nn.BatchNorm2d(64)
        #  Define two fully connected (linear) layers. These layers are typically used for the final classification or regression tasks.
        # A fully connected layer refers to a neural network in which each neuron applies a linear transformation to the input vector through a weights matrix. As a result, 
        # all possible connections layer-to-layer are present, meaning every input of the input vector influences every output of the output vector.
        self.fc1 = nn.Linear(7 * 7 * 64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        # Defines a softmax layer with dimension 1. This is used to convert raw model outputs into class probabilities.
        # The softmax function is used as the activation function in the output layer of neural network models 
        # that predict a multinomial probability distribution. 
        # That is, softmax is used as the activation function for multi-class classification problems 
        # where class membership is required on more than two class labels.
        self.softmax = nn.Softmax(dim=1)

    # Number of Layers
    # The number of layers in a neural network can depend on the complexity of your problem. 
    # Deeper networks can capture more complex patterns, but they are also more prone to overfitting. 
    # Start with a simple architecture and gradually increase complexity if necessary while monitoring validation performance.

    # Activation Functions
    # ReLU (Rectified Linear Unit) is a common choice for activation functions in hidden layers due to its simplicity and effectiveness. However, you can experiment with other activation functions like Sigmoid, Tanh, or Leaky ReLU to see what works best for your problem.
    
    # Connections Between Layers
    # The connections between layers are determined by the layer types and their arrangement in the forward method. Convolutional layers are typically followed by activation functions, and fully connected layers are commonly used in the final classification layers.    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 28x28->14x14

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.maxpool(x)  # 14x14->7x7

        # The feature maps are flattened into a 1D tensor using x.view.
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.softmax(x)

        return x
