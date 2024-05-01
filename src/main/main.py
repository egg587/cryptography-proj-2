import random
import warnings
from copy import deepcopy
import torch
import os

from cifar_utils import get_dataloader, plot_dataset, plot_history, torch_inference, train
from models import Fp32VGG11

# Grab current working directory
CURR_DIR = os.getcwd()
print(CURR_DIR)

dataset_name = "CIFAR_100"

device = "cuda" if torch.cuda.is_available() else "cpu"

param_c10 = {
    "output_size": 10,
    "batch_size": 128,
    "training": "fp32",
    "dataset_name": "CIFAR_10",
    "criterion": torch.nn.CrossEntropyLoss(),
    "accuracy_test": [],
    "accuracy_train": [],
    "loss_test_history": [],
    "loss_train_history": [],
    "dir": CURR_DIR + "/data/CIFAR_10",
    "seed": 727,
}
param_c100 = {
    "output_size": 100,
    "batch_size": 128,
    "training": "fp32",
    "dataset_name": "CIFAR_100",
    "criterion": torch.nn.CrossEntropyLoss(),
    "accuracy_test": [],
    "accuracy_train": [],
    "loss_test_history": [],
    "loss_train_history": [],
    "dir": CURR_DIR + "/data/CIFAR_100",
    "seed": 727,
}

# In this tutorial, we present the results of CIFAR-100.
if dataset_name == "CIFAR_100":
    param = param_c100
else:
    # If you want to use it for CIFAR-10, set `dataset_name` to "CIFAR_100"
    param = param_c10

print(f"Device Type: {device}")

################LOAD DATASET LOCALLY################

# Load CIFAR-100 or CIFAR-10 data-set according to `dataset_name`.
train_loader, test_loader = get_dataloader(param=param)

# Let’s visualize `n` images from CIFAR data-set.
#plot_dataset(test_loader, param)

################IMPORT TORCH MODEL AND FINE-TUNE################

torch.manual_seed(param["seed"])
random.seed(param["seed"])

# Instantiation of the custom VGG-11 network.
fp32_vgg = Fp32VGG11(param["output_size"]).to(device)

# Loading the pre-trained VGG-11 weights from torch.hub.
pretrained_weights = torch.hub.load(
    "pytorch/vision:v0.10.0",
    "vgg11",
    pretrained=True,
).state_dict()

# pretrained_weights = torch.load("./data/CIFAR_100/fp32/CIFAR_100_fp32_state_dict.pt")

# Caution:
# The `fp32_vgg.state_dict()` respects the same schema for the convolutional, ReLU and pooling
# layers (order/shape/naming) as the `pretrained_weights` dict, that's why it doesn't throw an error
fp32_vgg.load_state_dict(deepcopy(pretrained_weights), strict=False)
# We got an IncompatibleKeys warning because
# we deleted the classification layers of the original VGG-11 network.


# # We freeze all the layers.
for p in list(fp32_vgg.parameters()):
    p.requires_grad = False

# Set the `requires_grad` of the last layer at `True` to fine-tune it.
fp32_vgg.final_layer.weight.requires_grad = True
fp32_vgg.final_layer.bias.requires_grad = True


if dataset_name == "CIFAR_100":
    param["lr"] = 0.06
    param["epochs"] = 2
    param["gamma"] = 0.01
    param["milestones"] = [1]

elif dataset_name == "CIFAR_10":
    param["lr"] = 0.1
    param["epochs"] = 2
    param["gamma"] = 0.1
    param["milestones"] = [1, 3]

fp32_vgg = train(fp32_vgg, train_loader, test_loader, param, device=device)

# Secondly, fine-tuning all the layers.
for p in list(fp32_vgg.parameters()):
    p.requires_grad = True

if dataset_name == "CIFAR_100":
    param["lr"] = 0.0006
    param["epochs"] = 2
    param["gamma"] = 0.1
    param["milestones"] = [6]

elif dataset_name == "CIFAR_10":
    param["lr"] = 0.0006
    param["epochs"] = 2
    param["gamma"] = 0.1
    param["milestones"] = [4]

fp32_vgg = train(fp32_vgg, train_loader, test_loader, param, device=device)

# Don't need this line since it will save from the train() function
#torch.save(fp32_vgg.state_dict(), path_to_module)

################QAT TORCH MODEL AND FINE-TUNE################
#import torch
from cifar_utils import (
    fhe_compatibility,
    get_dataloader,
    mapping_keys,
    plot_baseline,
    plot_dataset,
    torch_inference,
    train,
)

# As we follow the same methodology for quantization aware training for CIFAR-10 and CIFAR-100.
# Let's import some generic functions.
from models import QuantVGG11
from torchvision import datasets

# Model Settings
bit = 5
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device Type: {device}")

# Setting the parameters for QAT with CIFAR100
param_c100 = {
    "output_size": 100,
    "batch_size": 128,
    "training": "quant",
    "dataset_name": "CIFAR_100",
    "dataset": datasets.CIFAR100,
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
    "lr": 6e-5,
    "seed": 727,
    "epochs": 1,
    "gamma": 0.01,
    "milestones": [3, 5],
    "criterion": torch.nn.CrossEntropyLoss(),
    "accuracy_test": [],
    "accuracy_train": [],
    "loss_test_history": [],
    "loss_train_history": [],
    "dir": CURR_DIR + "/data/CIFAR_100",
    "pre_trained_path": "fp32/CIFAR_100_fp32_state_dict.pt",
}

# Load CIFAR-100 data-set.
train_loader_c100, test_loader_c100 = get_dataloader(param=param_c100)

# Let’s visualize `n` images from CIFAR-100 data-set.
plot_dataset(test_loader_c100, param_c100)

quant_vgg = QuantVGG11(bit=bit, output_size=param_c100["output_size"])

# Load fp32 model we trained in the beginning
checkpoint = torch.load(
    f"{param_c100['dir']}/{param_c100['pre_trained_path']}", map_location=device
)

# Mapping Keys
# Need to make sure the model has the same network architecture and layer naming when loading it into its equivalent quantized network
quant_vgg = mapping_keys(checkpoint, quant_vgg, device)

# Measure Accuracy before QAT
# It's better than a random classifier.
acc_before_ft = torch_inference(quant_vgg, test_loader_c100, device=device)
param_c100["accuracy_test"].append(acc_before_ft)

print(f"Top 1 accuracy before fine-tuning = {acc_before_ft * 100:.4f}%")

# Testing FHE compatibility using fhe_compatibility function
# The user can either provide the entire train data-set or a smaller but representative subset.
# As each batch is shuffled and contains 128 samples, it's a potential subset.
data_calibration, _ = next(iter(train_loader_c100))

qmodel = fhe_compatibility(quant_vgg, data_calibration)

print(
    f"With {param_c100['dataset_name']}, the maximum bit-width in the circuit = "
    f"{qmodel.fhe_circuit.graph.maximum_integer_bit_width()}"
)

# Fine-tuning using QAT with Brevitas
quant_vgg = train(quant_vgg, train_loader_c100, test_loader_c100, param_c100, device=device)

# Plot fine-tuning results
plot_baseline(param_c100, test_loader_c100, device)
