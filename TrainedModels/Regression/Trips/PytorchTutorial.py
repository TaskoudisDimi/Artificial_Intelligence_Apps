
# Pytorch

# #### Torch examples
# data =[[1,2,3,4], [3,4,5,6]]

# data_tensor = torch.tensor(data)
# print(len(data_tensor))


# x_rand = torch.rand_like(data_tensor, dtype=torch.float)
# print(x_rand)

# tensor1 = torch.randn(3,4)
# tensor2 = torch.randn(4)

# result = torch.matmul(tensor1, tensor2)

# print(result)


##############################################################
### Simple CNN model for Iris dataset
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torch.nn.functional as F
# from torch.utils.data import DataLoader, TensorDataset
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from torchvision.datasets import ImageFolder
# import matplotlib.pyplot as plt 

# device = ""
# if torch.cuda.is_available():
#     device = "cuda"
# else:
#     device = "cpu"

# print(device)
    
# iris = load_iris()
# X = iris.data
# y = iris.target

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# X_train_Tensor = torch.tensor(X_train, dtype=torch.float32)
# y_train_Tensor = torch.tensor(y_train, dtype=torch.int64)
# X_test_Tensor = torch.tensor(X_test, dtype=torch.float32)
# y_test_Tensor = torch.tensor(y_test, dtype=torch.int64)


# train_dataset = TensorDataset(X_train_Tensor, y_train_Tensor)
# train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


# class CNN(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(CNN, self).__init__()
#         self.fc1 = nn.Linear(input_size, hidden_size)
#         self.fc2 = nn.Linear(hidden_size, num_classes)

#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
    

# input_size = 4
# hidden_size = 64
# num_classes = 3
# model = CNN(input_size, hidden_size, num_classes)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.001)


# num_epochs = 100
# train_losses = []

# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         # When you're training a neural network using gradient-based optimization algorithms 
#         # like stochastic gradient descent (SGD) or Adam, you need to update the model's weights 
#         # based on the gradients of the loss with respect to those weights. Before calculating 
#         # these gradients in a new forward pass, it's crucial to zero out the gradients from the previous pass.
#         optimizer.zero_grad()  # Zero out gradients from the previous iteration
#         outputs = model(inputs) #call forward
#         loss = criterion(outputs, labels)#compute loss
#         loss.backward() #perform backpropagation to update the model's weights
#         optimizer.step() # Update model weights based on gradients
#         running_loss += loss.item()
    
#     avg_loss = running_loss / len(train_loader)
#     train_losses.append(avg_loss)
#     print(f'Average loss is {avg_loss:.4f}')
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')




# plt.plot(range(1, num_epochs + 1), train_losses, label='Training Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.title('Train Loss Curve')
# plt.grid(True)
# plt.show()



# model.eval()
# # During inference or evaluation, you often don't need to compute gradients, 
# and it can be computationally expensive and unnecessary. 
# # Temporarily disable gradient computation for a block of code.
# #  This can lead to faster execution and reduced memory usage when you don't need gradients.
# with torch.no_grad():
#     X_test_Tensor = X_test_Tensor.to(device)
#     outputs = model(X_test_Tensor)
#     _, predicted = torch.max(outputs, 1)
#     accuracy = (predicted == y_test_Tensor).float().mean()
#     print(f'Accuracy on the test data: {accuracy.item():.2%}')



##############################################################

# ## Simple CNN for MNIST data 
# import torch
# import torch.nn as nn
# import torch.optim as optim
# import torchvision
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader, TensorDataset
# import torch.nn.functional as F


# # Data preprocessing and loading


# # Hyperparameters
# batch_size = 64
# learning_rate = 0.001
# num_epochs = 10


# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform, download=True)

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)




# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.fc1 = nn.Linear(in_features=64 * 7 * 7, out_features=128)
#         self.fc2 = nn.Linear(in_features=128, out_features=10)

#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))  # Use F.relu
#         x = self.pool(F.relu(self.conv2(x)))  # Use F.relu
#         x = x.view(-1, 64 * 7 * 7)
#         x = F.relu(self.fc1(x))  # Use F.relu
#         x = self.fc2(x)
#         return x



# model = CNN()
# optim = optim.Adam(model.parameters(), lr=learning_rate)
# criterion = nn.CrossEntropyLoss()



# train_losses = []
# for epoch in range(num_epochs):
#     running_loss = 0.0
#     for inputs, labels in train_loader:
#         optim.zero_grad()  # Zero out gradients from the previous iteration
#         outputs = model(inputs) #call forward
#         loss = criterion(outputs, labels)#compute loss
#         loss.backward() #perform backpropagation to update the model's weights
#         optim.step() # Update model weights based on gradients
#         running_loss += loss.item()
    
#     avg_loss = running_loss / len(train_loader)
#     train_losses.append(avg_loss)
#     print(f'Average loss is {avg_loss:.4f}')
#     print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')


# # Test the model
# correct = 0
# total = 0
# with torch.no_grad():
#     for data in test_loader:
#         inputs, labels = data
#         outputs = model(inputs)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()

# print(f'Accuracy on the test images: {(100 * correct / total):.2f}%')

##############################################################


# ## Simple CNN for Fashion MNIST data 
# import torch
# from torchvision.transforms import ToTensor
# from torchvision import datasets
# from torch.utils.data import DataLoader
# from torch import nn



# training_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
# test_data = datasets.FashionMNIST(root='data', train=False,download=True, transform=ToTensor())

# train_dataloader = DataLoader(training_data, batch_size=64)
# test_dataloader = DataLoader(test_data, batch_size=64)


# class NeuralNetwork(nn.Module):
#     def __init__(self):
#         super().__init__()
#         # We initialize the nn.Flatten layer to convert each 2D 28x28 image into a contiguous array of 784 pixel values
#         self.flatten = nn.Flatten()
#         # The linear layer is a module that applies a linear transformation on the input using its stored weights and biases.
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28*28,512),
#             # Non-linear activations are what create the complex mappings between 
#             # the model’s inputs and outputs. They are applied after linear transformations 
#             # to introduce nonlinearity, helping neural networks learn a wide variety of phenomena.
#             # In this model, we use nn.ReLU between our linear layers, but there’s other activations to introduce non-linearity in your model.
#             nn.ReLU(),
#             nn.Linear(512,512),
#             nn.ReLU(),
#             nn.Linear(512,10)
#             )
    
#     # nn.Sequential is an ordered container of modules. The data is passed through all 
#     # the modules in the same order as defined. You can use sequential containers to 
#     # put together a quick network like seq_modules.
    
#     # The last linear layer of the neural network returns logits - raw values 
#     # in [-infty, infty] - which are passed to the nn.Softmax module. 
#     # The logits are scaled to values [0, 1] representing the model’s predicted probabilities 
#     # for each class. dim parameter indicates the dimension along which the values must sum to 1.
    
    
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
  
    
# device = "cpu"
# model = NeuralNetwork().to(device)
# print(model)




# def train_loop(dataloader, model, loss_fn, optimizer):
#     size = len(dataloader.dataset)
#     # Set the model to training mode - important for batch normalization and dropout layers
#     # Unnecessary in this situation but added for best practices
#     model.train()
#     for batch, (X, y) in enumerate(dataloader):
#         # Compute prediction and loss
#         pred = model(X)
#         loss = loss_fn(pred, y)

#         # Backpropagation
#         loss.backward()
#         optimizer.step()
#         optimizer.zero_grad()

#         if batch % 100 == 0:
#             loss, current = loss.item(), (batch + 1) * len(X)
#             print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


# def test_loop(dataloader, model, loss_fn):
#     # Set the model to evaluation mode - important for batch normalization and dropout layers
#     # Unnecessary in this situation but added for best practices
#     model.eval()
#     size = len(dataloader.dataset)
#     num_batches = len(dataloader)
#     test_loss, correct = 0, 0

#     # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
#     # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
#     with torch.no_grad():
#         for X, y in dataloader:
#             pred = model(X)
#             test_loss += loss_fn(pred, y).item()
#             correct += (pred.argmax(1) == y).type(torch.float).sum().item()

#     test_loss /= num_batches
#     correct /= size
#     print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")





# loss_fn = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=0.001)

# epochs = 10
# for t in range(epochs):
#     print(f"Epoch {t+1}\n-------------------------------")
#     train_loop(train_dataloader, model, loss_fn, optimizer)
#     test_loop(test_dataloader, model, loss_fn)
# print("Done!")

##############################################################




# import torch
# import torchvision.models as models


# # Save/Load Weights
# model = models.vgg16(weights='IMAGENET1K_V1')
# torch.save(model.state_dict(), 'model_weights.pth')

# model = models.vgg16() # we do not specify ``weights``, i.e. create untrained model
# model.load_state_dict(torch.load('model_weights.pth'))
# model.eval()


# # Save/Load Models with Shapes
# torch.save(model, 'model.pth')
# model = torch.load('model.pth')



##############################################################
# Torch Tutorial


import torch
import numpy as np


# x = torch.empty(2,2,2,3)



# x = torch.zeros(2,2)
# print(x.size())


# y = torch.ones(2,2, dtype=torch.double)
# print(y.size())

# print(y.type)
# print(y.size())

# z = torch.tensor([2.5, 0.1])
# print(z)


# x = torch.rand(2,2)
# y = torch.rand(2,2)
# z = x - y 

# z = torch.add(x, y)
# z = torch.sub(x, y)
# z = torch.mul(x, y)
# z = torch.div(x, y)

# print(z)
# print(z.size())




# x = torch.rand(4,4)
# print(x)
# y = x.view(2,8)
# print(y)


# From torch to numpy
# a = torch.ones(5)
# print(a)
# b = a.numpy()
# print(b)

# a.add_(1)
# print(a)
# print(b)


# # From numpy to torch
# a = np.ones(5)
# print(a)
# b = torch.from_numpy(a)
# print(b)

# a += 1
# print(a)
# print(b)




# if torch.cuda.is_available():
#     print("Cuda")
# else:
#     print("No Cuda")



# requires_grad. The requires_grad attribute tells autograd to track your operations. 
# So if you want PyTorch to create a graph corresponding to these operations, 
# you will have to set the requires_grad attribute of the Tensor to True. 
# There are 2 ways in which it can be done, either by passing it as an argument in torch.

# x = torch.randn(3, requires_grad=True)
# print(x)

# y = x+2
# print(y)

# z = y*y*2
# z = z.mean()
# print(z)
# z.backward()



# x.requires_grad_(False)
# print(x)


# y = x.detach()
# print(y)


# with torch.no_grad():
#     y = x +2
#     print(y)




# Backpropagation proccess
x = torch.tensor(1.0)
y = torch.tensor(2.0)

w = torch.tensor(1.0, requires_grad=True)
print(w)

# Forward pass and compute the loss
y_hat = w * x
print(y_hat)
loss = (y_hat - y) ** 2
print(loss)


# Backward pass
# The backward() method in Pytorch is used to calculate the gradient during the backward pass 
# in the neural network. If we do not call this backward() method then gradients are not 
# calculated for the tensors. The gradient of a tensor is calculated for 
# the one having requires_grad is set to True
loss.backward()
print(w.grad)


# Update weights
# Next forward and backwards









































