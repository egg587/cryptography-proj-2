import time

import numpy as np
import torch
import torch.utils
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from concrete.ml.torch.compile import compile_torch_model

# And some helpers for visualization.

import matplotlib.pyplot as plt


# Load the data set and visualize it
X, y = load_digits(return_X_y=True)

# The sklearn Digits dataset, though it contains digit images, keeps these images in vectors.
# We need to reshape them to 2D to visualize them. The images are 8x8 pixels in size and monochrome.
X = np.expand_dims(X.reshape((-1, 8, 8)), axis=1)

nplot = 4
fig, axes = plt.subplots(nplot, nplot, figsize=(6, 6))
for i in range(0, nplot):
    for j in range(0, nplot):
        axes[i, j].imshow(X[i * nplot + j, ::].squeeze())
plt.show()

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, shuffle=True, random_state=42)


# Define the Neural Network model
"""
Since the accumulator bit width in FHE is small, we prune the convolutional filters to limit the number of connections per neuron.

Neural network pruning is the process by which the synapses of individual neurons in a layer are forced to have a weight equal to zero. This basically eliminates
them from the computation and thus they do not increase teh accumulator bit width. It has been shown that neural networks can maintain their accuracy with a degree of pruning that can even
exceed 70% for some over-parameterized networks such as VGG-16 or large ResNets.

See: https://arxiv.org/pdf/2003.03033.pdf, Figure 8 in Section 7.2, for an evaluation on the simple pruning method used in this example.
"""

class TinyCNN(nn.Module):
    """A very small CNN to classify the sklearn digits dataset."""

    def __init__(self, n_classes) -> None:
        """Construct the CNN with a configurable number of output classes."""
        super().__init__()

        # This network has a total complexity of 1216 MAC
        self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=0)
        self.conv2 = nn.Conv2d(8, 16, 3, stride=2, padding=0)
        self.conv3 = nn.Conv2d(16, 32, 2, stride=1, padding=0)
        self.fc1 = nn.Linear(32, n_classes)

    def forward(self, x):
        """Run inference on the tiny CNN, apply the decision layer on the reshaped conv output."""
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        x = torch.relu(x)
        x = x.flatten(1)
        x = self.fc1(x)
        return x


# Train the CNN
"""
Note that the training code from for quantization aware training is the same as it would be for floating point training.
Indeed, the Brevitas layers used in the CNN class will handle quantization during training. We train the network for varying weights
and activation bit-width, to find an FHE compatible configuration.
"""

torch.manual_seed(42)

def train_and_epoch(net, optimizer, train_loader):
    # Cross Entropy Loss for classification when not using a softmax layer in the network
    loss = nn.CrossEntropyLoss()
    
    net.train()
    avg_loss = 0
    for data, target in train_loader:
        #print(f"Input data shape: {data.shape}")  # Check input data shape
        optimizer.zero_grad()
        output = net(data)
        loss_net = loss(output, target.long())
        loss_net.backward()
        optimizer.step()
        avg_loss += loss_net.item()
    
    return avg_loss / len(train_loader)

# Create the tiny CNN with 10 output classes
N_EPOCHS = 150

# Create a train data loader
print("Creating training data loaders...")
train_dataset = TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train))
train_dataloader = DataLoader(train_dataset, batch_size=64)
print("Training data loader created.")

# Create a test data loader to supply batches for network evaluation (test)
print("Creating test data loaders...")
test_dataset = TensorDataset(torch.Tensor(X_test), torch.Tensor(y_test))
test_dataloader = DataLoader(test_dataset)
print("Test data loader created.")

# Train the network with Adam, output the test set accuracy every epoch
print("Creating the network...")
net = TinyCNN(10)
losses_bits = []
optimizer = torch.optim.Adam(net.parameters())
print("Training the network...")
epoch = 1
for _ in tqdm(range(N_EPOCHS), desc="Training"):
    loss = train_and_epoch(net, optimizer, train_dataloader)
    # display loss and epoch count
    # print(f"Loss: {loss}")
    # print(f"Epoch: {epoch}")
    losses_bits.append(loss)
    epoch += 1
print("Training complete.")

# Plot the loss
# fig = plt.figure(figsize=(8, 4))
# plt.plot(losses_bits)
# plt.xlabel("Epoch")
# plt.ylabel("Cross Entropy Loss")
# plt.title("Training set loss during training")
# plt.grid(True)
# plt.show()

# Test the torch network in fp32
def test_torch(net, test_loader):
    """Test the network: measure accuracy on the test set."""

    # Freeze normalization layers
    net.eval()

    all_y_pred = np.zeros((len(test_loader.dataset)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader.dataset)), dtype=np.int64)

    # Iterate over the batches
    idx = 0
    for data, target in test_loader:
        # Accumulate the ground truth labels
        endidx = idx + target.shape[0]
        all_targets[idx:endidx] = target.numpy()

        # Run forward pass and get the predicted class id
        output = net(data).argmax(1).detach().numpy()
        all_y_pred[idx:endidx] = output

        idx += target.shape[0]
    
    # Print out the accuracy as a percentage
    n_correct = np.sum(all_targets == all_y_pred)
    print(
        f"Test accuracy for fp32 weights and activations:\n{n_correct / len(test_loader) * 100:.2f}%"
    )

test_torch(net, test_dataloader)

# Define the Concrete ML testing function
"""
We introduce the test_with_concrete function which allows us to test a Concrete ML model in one of two modes:
- in FHE
- in the clear, using simulated FHE execution
Note that it is trivial to toggle between the two modes.
"""

def test_with_concrete(quantized_module, test_loader, use_sim):
    """Test a neural network that is quantized and compiled with Concrete ML."""

    # Casting the inputs into int64 is recommended
    all_y_pred = np.zeros((len(test_loader)), dtype=np.int64)
    all_targets = np.zeros((len(test_loader)), dtype=np.int64)

    # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
    idx = 0
    for data, target in tqdm(test_loader):
        data = data.numpy()
        target = target.numpy()

        fhe_mode = "simulate" if use_sim else "execute"

        # Quantize the inputs and cast to appropriate data type
        y_pred = quantized_module.forward(data, fhe=fhe_mode)

        endidx = idx + target.shape[0]

        # Accumulate the ground truth labels
        all_targets[idx:endidx] = target

        # Get the predicted class id and accumulate the predictions
        y_pred = np.argmax(y_pred, axis=1)
        all_y_pred[idx:endidx] = y_pred

        # Update the index
        idx += target.shape[0]
    
    # Compute and report results
    n_correct = np.sum(all_targets == all_y_pred)

    return n_correct / len(test_loader)


# Test the network using Simulation
"""
Note that this is not a test in FHE. The simulated FHE mode gives insight about the impact of FHE execution on the accuracy.
The torch neural network is converted to FHE by Concrete ML using a dedicated function, compile_torch_model.
"""

n_bits = 6

q_module = compile_torch_model(net, X_train, rounding_threshold_bits=n_bits, p_error=0.1)

start_time = time.time()
accuracy = test_with_concrete(
    q_module, 
    test_dataloader, 
    use_sim=True
)
sim_time = time.time() - start_time

print(f"Simulated FHE execution for {n_bits} bit network accuracy: {accuracy * 100:.2f}%")

# Generate Keys
t = time.time()
q_module.fhe_circuit.keygen()
print(f"Keygen time: {time.time() - t:.2f}s")

# Execute in FHE on encrypted data
# Run inference in FHE on a single encrypted example
mini_test_dataset = TensorDataset(torch.Tensor(X_test[0:100, :]), torch.Tensor(y_test[0:100]))
mini_test_dataloader = DataLoader(mini_test_dataset)

t = time.time()
accuracy_test = test_with_concrete(
    q_module, 
    mini_test_dataloader, 
    use_sim=False
)

elapsed_time = time.time() - t
time_per_inference = elapsed_time / len(mini_test_dataloader)
accuracy_percentage = accuracy_test * 100
print(f"Time per inference in FHE: {time_per_inference:.2f}s with {accuracy_percentage:.2f}% accuracy")


# Conclusion
"""
In this example, a simple CNN model is trained with torch and reach 98% accuracy in clear. The model is then converted to FHE and evaluated over 100 samples in FHE.

The model in FHE achieves the same accuracy as the original torch model with a FHE execution time of 2.9 seconds per image.
"""

