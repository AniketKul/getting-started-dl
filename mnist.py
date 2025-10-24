# Goal: Training a deep learning model to correctly classify hand-written digits.

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
import matplotlib.pyplot as plt

# visualization tools
import torchvision
import torchvision.transforms.v2 as transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.is_available()

train_set = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True
)

valid_set = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True
)

print(f"Training set size: {len(train_set)}")
print(f"Validation set size: {len(valid_set)}")

"""
If a vector is a 1-dimensional array, and a matrix is a 2-dimensional array, 
a tensor is an n-dimensional array representing any number of dimensions. 
Most modern neural network frameworks are powerful tensor processing tools.
"""

trans = transforms.Compose([transforms.ToTensor()])
x_0_tensor = trans(x_0)
print(f"Tensor shape: {x_0_tensor.shape}")
print(f"Tensor type: {x_0_tensor.dtype}")
print(f"Tensor min: {x_0_tensor.min()}")
print(f"Tensor max: {x_0_tensor.max()}")
print(f"Tensor size: {x_0_tensor.size()}")

# By default, a tensor is processed with a CPU.
# To move it to a GPU, we can use the .cuda method.
x_0_gpu.device = x_0_tensor.cuda()
x_0_gpu.device
print(f"Tensor device: {x_0_gpu.device}")

# We can also move a tensor back to the CPU using the .cpu method.
x_0_tensor = x_0_tensor.cpu()
print(f"Tensor device: {x_0_tensor.device}")

# We can also convert a tensor to a numpy array using the .numpy method.
x_0_tensor = x_0_tensor.numpy()
print(f"Tensor type: {type(x_0_tensor)}")

# The .cuda method will fail if a GPU is not recognized by PyTorch. 
# In order to make our code flexible, we can send our tensor to the device we identified at the 
# start of this notebook. This way, our code will run much faster if a GPU is available, 
# but the code will not break if there is no available GPU.

x_0_tensor.to(device).device

# Sometimes, it can be hard to interpret so many numbers. 
# Thankfully, TorchVision can convert C x H x W tensors back into a 
# PIL image with the to_pil_image function.

image = F.to_pil_image(x_0_tensor)
plt.imshow(image, cmap='gray')
plt.show()

""" 1.4 Preparing the data for training """

trans = transforms.Compose([transforms.ToTensor()])

train_set.transform = trans
valid_set.transform = trans

"""
If a dataset is a deck of flash cards, 
a DataLoader defines how we pull cards from the deck to train an AI model. 
We could show our models the entire dataset at once. Not only does this take 
a lot of computational resources, but research shows using a smaller batch of 
data is more efficient for model training.
"""

batch_size = 32
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size)

""" 1.5 Creating the Model """

layers = []
test_matrix = torch.tensor(
    [[1, 2, 3],
     [4, 5, 6],
     [7, 8, 9]]
)
print(test_matrix)
nn.Flatten()(test_matrix)

# Currently, the Flatten layer sees three vectors as opposed to one 2d matrix. 
# To fix this, we can "batch" our data by adding an extra pair of brackets. 
# Since `test_matrix` is now a tensor, we can do that with the shorthand below. 
# `None` adds a new dimension where `:` selects all the data in a tensor.

batch_test_matrix = test_matrix[None, :]
print(batch_test_matrix)
nn.Flatten()(batch_test_matrix)

layers = [
    nn.Flatten()
]

""" 1.5.2 The Input Layer """

# Our first layer of neurons connects our flattened image to the rest of our model. 
# To do that, we will use a Linear layer. This layer will be densely connected, 
# meaning that each neuron in it, and its weights, will affect every neuron in the next layer.
input_size = 1 * 28 * 28
layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
]

""" 1.5.3 The Hidden Layer """

# Now we will add an additional densely connected linear layer. 
# We will cover why adding another set of neurons can help improve 
# learning in the next lesson. Just like how the input layer needed to 
# know the shape of the data that was being passed to it, a hidden layer's Linear needs to know the shape of the data being passed to it. Each neuron in the previous layer will compute one number, so the number of inputs into the hidden layer is the same as the number of neurons in the previous later.

layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 512),  # Hidden
    nn.ReLU(),  # Activation for hidden
]

""" 1.5.4 The Output Layer """

# Finally, we will add an output layer. In this case, since the 
# network is to make a guess about a single image belonging to 1 of 10 
# possible categories, there will be 10 outputs. Each output is assigned 
# a neuron. The larger the value of the output neuron compared to the other 
# neurons, the more the model predicts the input image belongs to the 
# output neuron's assigned class.

n_classes = 10
layers = [
    nn.Flatten(),
    nn.Linear(input_size, 512),  # Input
    nn.ReLU(),  # Activation for input
    nn.Linear(512, 512),  # Hidden
    nn.ReLU(),  # Activation for hidden
    nn.Linear(512, n_classes)  # Output
]

""" 1.5.5 Compiling the Model """

# A Sequential model expects a sequence of arguments, not a list, 
# so we can use the * operator to unpack our list of layers into a sequence. 
# We can print the model to verify these layers loaded correctly.

model = nn.Sequential(*layers)

# Much like tensors, when the model is first initialized, it will be processed on a CPU. To have it process with a GPU, we can use `to(device)`.

model.to(device)

# To check which device a model is on, we can check which device the model parameters are on. Check out this [stack overflow](https://stackoverflow.com/questions/58926054/how-to-get-the-device-type-of-a-pytorch-module-conveniently) post for more information.

next(model.parameters()).device
model = torch.compile(model)

""" 1.6 Training the Model """

# Training a model with data is often also called "fitting a model to data." 
# Put another way, it highlights that the shape of the model changes over 
# time to more accurately understand the data that it is being given.

""" 1.6.1 Loss and Optimization """

# Just like how teachers grade students, we need to provide the model a 
# function in which to grade its answers. This is called a `loss function`. 
# We will use a type of loss function called [CrossEntropy](https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html) 
# which is designed to grade if a model predicted the correct category from a group of categories.

loss_function = nn.CrossEntropyLoss()

# Next, we select an `optimizer` for our model. If the `loss_function` provides a grade, the optimizer tells the model how to learn from this grade to do better next time.

optimizer = Adam(model.parameters())

""" 1.6.2 Calculating Accuracy """

