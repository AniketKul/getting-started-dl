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

## Preparing data for training

trans = transforms.Compose([transforms.ToTensor()])
