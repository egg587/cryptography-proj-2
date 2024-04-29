
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import time
from tqdm import tqdm
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

from concrete.ml.torch.compile import compile_torch_model

#%matplotlib inline

# Define the Neural Network model

class ImageClass(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def imshow(self, img):
        img = img / 2 + 0.5     # unnormalize
        npimg = img.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def model_training_func(self, num_of_epochs, trainloader, compile_params):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(num_of_epochs):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):

                inputs, labels = data
                #inputs, labels = inputs.to(device), labels.to(device)

                outputs = self.forward(inputs)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 2000 == 1999:
                    print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                    running_loss = 0.0
        
        print('Training Completed')

        # Compile the model using Concrete ML
        n_bits = compile_params.get('n_bits', 6)
        p_error = compile_params.get('p_error', 0.1)

        # Create a batch of inputs
        inputs_example, _ = next(iter(trainloader))
        q_module = compile_torch_model(self, inputs_example, rounding_threshold_bits=n_bits, p_error=p_error)

        return q_module

        
    def test_with_concrete(self, quantized_module, test_loader, use_sim):
        """Test a neural network that is quantized and compiled with Concrete ML."""

        # Casting the inputs into int64 is recommended
        all_y_pred = []
        all_targets = []

        # Iterate over the test batches and accumulate predictions and ground truth labels in a vector
        for data, targets in tqdm(test_loader):
            # print(f"Printing data: {data}")
            # print(f"Printing targets: {targets}")

            if data.requires_grad:
                data = data.detach()

            if data.is_cuda:
                data = data.cpu()

            fhe_mode = "simulate" if use_sim else "execute"

            # Quantize the inputs and cast to appropriate data type
            data = data.numpy()
            #data = data.astype(np.int64)
            predictions = quantized_module.forward(data, fhe=fhe_mode)

            # Convert predictions and targets to numpy arrays
            targets = targets.numpy()

            # Accumulate the ground truth labels
            all_y_pred.extend(np.argmax(predictions, axis=1))
            all_targets.extend(targets)
        
        # Convert accumulated lists to numpy arrays
        all_y_pred = np.array(all_y_pred)
        all_targets = np.array(all_targets)

        # Compute Accuracy
        accuracy = np.mean(all_y_pred == all_targets)

        return accuracy


if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load a dataset
    batch_size = 2
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Instantiate the model
    model = ImageClass()
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model.to(device)

    # Train model and compile with Concrete ML
    num_epochs = 1
    compile_params = {'n_bits': 6, 'p_error': 0.1}
    q_module = model.model_training_func(num_epochs, trainloader, compile_params)

    # Evaluate the model using Concrete ML simulation
    start_time = time.time()
    accuracy = model.test_with_concrete(
        q_module,
        testloader,
        use_sim=True
    )
    sim_time = time.time() - start_time

    print(f"Simulated FHE execution for {compile_params['n_bits']} bit network accuracy: {accuracy * 100:.2f}%")
    print(f"Simulation time: {sim_time:.2f} seconds")

    print(f"Running testing...")

    # Generate Keys
    t = time.time()
    q_module.fhe_circuit.keygen()
    print(f"Keygen time: {time.time() - t:.2f}s")

    print(f"Length of test set: {len(testloader)}")

    t = time.time()
    accuracy_test = model.test_with_concrete(
        q_module, 
        testloader, 
        use_sim=False
    )

    elapsed_time = time.time() - t
    time_per_inference = elapsed_time / len(testloader)
    accuracy_percentage = accuracy_test * 100
    print(f"Time per inference in FHE: {time_per_inference:.2f}s with {accuracy_percentage:.2f}% accuracy")
