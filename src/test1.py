
import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

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

    def model_training_func(self, num_of_epochs):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

        for epoch in range(num_of_epochs):

            running_loss = 0.0
            for i, data in enumerate(trainloader, 0):

                inputs, labels = data

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
        return 0


if __name__ == "__main__":

    transform = transforms.Compose([transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load a dataset
    batch_size = 4
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                            shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = ImageClass()

    # Train model on 5 epochs
    num_epochs = 5
    model.model_training_func(num_epochs)

    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # print images
    model.imshow(torchvision.utils.make_grid(images))
    print('Ground Truth Labels: ', ' '.join(f'{classes[labels[j]]:5s}' for j in range(4)))

    PATH = './cifar_net.pth'
    torch.save(model.state_dict(), PATH)

    # model = nn.Sequential(
    #       nn.Conv2d(3,6,5),
    #       nn.ReLU(),
    #       nn.MaxPool2d(2, 2),
    #       nn.Conv2d(6, 16, 5),
    #       nn.ReLU(),
    #       nn.MaxPool2d(2, 2),
    #       nn.Flatten(),
    #       nn.Linear(16 * 5 * 5, 120),
    #       nn.ReLU(),
    #       nn.Linear(120, 84),
    #       nn.ReLU(),
    #       nn.Linear(84, 10))
    model.load_state_dict(torch.load(PATH))

    outputs = model(images)
    _, predicted = torch.max(outputs, 1)

    print('Predicted Labels by the model: ', ' '.join(f'{classes[predicted[j]]:5s}'
                                for j in range(4)))

    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            
            outputs = model(images)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Model Accuracy on test data : {100 * correct // total} %')

    correct_pred = {classname: 0 for classname in classes}
    total_pred = {classname: 0 for classname in classes}

    with torch.no_grad():
        for data in testloader:
            images, labels = data
            outputs = model(images)
            _, predictions = torch.max(outputs, 1)
            for label, prediction in zip(labels, predictions):
                if label == prediction:
                    correct_pred[classes[label]] += 1
                total_pred[classes[label]] += 1



    for classname, correct_count in correct_pred.items():
        accuracy = 100 * float(correct_count) / total_pred[classname]
        print(f'Accuracy for the class: {classname} is {accuracy} %')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)

    model.to(device)
    inputs, labels = data
    inputs, labels = inputs.to(device), labels.to(device)


    # Compile with Concrete
    # n_bits = 6
    # q_module = compile_torch_model(model, X_train, rounding_threshold_bits=n_bits, p_error=0.1)




    


