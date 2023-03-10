import torch
import torch.nn as nn
from torchvision import models
import wandb
import os
import math
import json
import matplotlib.pyplot as plt
from data_loader import get_dataloaders
from random import sample
from functools import reduce
from torch import stack

from config import *

# example model resnet, maybe add a layer to match our image sizes in the beginning
device = torch.device(train_conf["device"])

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

# Adapt model to 6 class output
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(512, len(classes))
model = model.float()
model = model.to(device)

# Set performance criterion
criterion = torch.nn.CrossEntropyLoss()

# Get Dataloaders
dataloader_dict = get_dataloaders()

# Predefined indices of specific images to test
statistic_idx = sample(range(len(dataloader_dict["train"])), 200)
json_data = {}


def get_module_by_name(module, access_string):
    names = access_string.split(sep='.')
    return reduce(getattr, names, module)


def computeStatistics(model, dataloader, indices, modules):
    # Register hooks
    for nam, mod in model.named_modules():
        print(nam)
        if nam in modules:
            module = get_module_by_name(model, nam)
            module.register_forward_hook(getLayerName(nam))

    # Create custom batch
    batch = {"image": [], "label": []}
    for idx in indices:
        image, label = dataloader[idx]
        batch["image"].append(image)
        batch["label"].append(label)

    batch["image"] = stack(batch["image"])
    batch["label"] = stack(batch["label"])

    batch["image"] = batch["image"].to(device)
    model(batch["image"])


def getLayerName(name):
    # Layer shape is CONV so it should be B, C, H, W
    def statisticHook(model, input, output):
        input = input[0]
        output_shape = list(output.shape)

        # Select every axis except the first one that includes batches
        output_dim_axis = [i for i in range(len(output_shape))][1:]

        # Area of the filters
        output_area = math.prod(output_shape[1:])

        # Average over all dimensions except batch
        output_averages = torch.divide(torch.count_nonzero(torch.greater_equal(output, 0), dim=output_dim_axis), output_area)

        # Average over batches
        output_nonpositive_avg = torch.mean(output_averages)

        # Calculate mean over spatial dims
        output_mean_tensor = torch.mean(output, dim=output_dim_axis[1:])

        # Covariance
        batch_mean = torch.mean(output_mean_tensor, dim=0)
        covariance = torch.zeros(size=(output_shape[1], output_shape[1]))
        covariance = covariance.to(device)
        for o in output_mean_tensor:
            covariance += torch.dot(o, o.T) - torch.dot(batch_mean, batch_mean.T)
        covariance = torch.divide(covariance, 200)

        # Calculate eigenvalues
        w, lamb = torch.eig(covariance, eigenvectors=True)
        w, indices = torch.sort(w, dim=0, descending=True) # Complex sort

        # Plot norm of eigenvalues
        norms, indices = torch.sort(torch.norm(w, dim=1), descending=True) # Norm sort
        norms = norms.to('cpu').detach().numpy()
        plt.plot(range(0, len(norms)), norms, 'xb-')
        plt.savefig(os.path.join(data_root, name + '.jpg'))

        json_data[name] = {"avg_nonpositive": output_nonpositive_avg, "avg": output_mean_tensor, "cov": covariance, "eigvals": w}
    return statisticHook


def saveCheckpoint(model, optimizer, filename):
    """
    saves current model and optimizer step

    model: nn.model
    optimizer: torch.optim.optimzer class
    filename: string
    """
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
    torch.save(checkpoint, filename)
    print("checkpoint saved")
    return


def loadCheckpoint(model, optimizer, path):
    """
    loads mode and optimzer for further training
    model: nn.model
    optimizer: torch.optim.optimzer class
    path: string
    return: list of optimizer and model

    """
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    print("checkpoint loaded")
    return


#################


def trainLoop(dataLoader, lossFunction, model, params, loadCheckpoint, WandB, pathOrigin):
    """
    Trains a deep learning model with a variant of gradient decent

    Parameters
    ----------
    dataLoader: torch dataLoader object
    lossFunction: torch.nn function
    model: torch.nn object
    params: dict
    loadCheckpoint: boolean
    WandB: boolean
    pathOrigin: string

    Returns: _
    -------
    """
    global device

    # WandB
    if WandB:
        wandb.init(
            # set the wandb project where this run will be logged
            project=params["modelName"],
            # track params and run metadata
            config={
                "learning_rate": params["learningRate"],
                "architecture": params["modelName"],
                "epochs": params["epochs"],
            },
        )

    # optimizer
    if params["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["learningRate"],
            weight_decay=params["weightDecay"],
        )

    # load model
    if loadCheckpoint:
        # get into folder
        model_name = str(params["modelName"])
        loadCheckpoint(model, optimizer, os.path.join(pathOrigin, "models", model_name))

    # start training
    train_count = 0
    model.train()
    runningLoss = 0
    for i in range(params["epochs"]):
        batch_idx_range = range(len(dataLoader) // params["batch_size"])
        for batch_idx in batch_idx_range:
            sample = dataLoader.get_batch(params["batch_size"], batch_idx)

            input_data, labels = sample["image"], sample["label"]
            #input_data = torch.nn.functional.normalize(input_data)  # normalize image
            input_data = input_data.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # calculate loss + backprop
            pred = model(input_data)
            loss = lossFunction(pred, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.item()
            train_count += 1
            print("Epoch: ", i, "current loss: ", runningLoss / train_count)

            ### implement early stopping here #####

            ########################################

            # save checkpoint at each end of epoch
        model_name = str(params["modelName"])
        saveCheckpoint(model, optimizer, os.path.join(pathOrigin, "models", model_name))

    return


trainLoop(
    dataloader_dict["train"],
    lossFunction=criterion,
    model=model,
    params=train_conf,
    loadCheckpoint=False,
    WandB=False,
    pathOrigin=data_root,
)

modules = ['conv1', 'layer1.0.conv1', 'layer1.1.conv1', 'layer2.0.conv1']
computeStatistics(model, dataloader_dict["train"], statistic_idx, modules)

