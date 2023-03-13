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

# Get dataloaders
dataloader_dict = get_dataloaders()

json_data = {}


def get_module_by_name(module, access_string):
    names = access_string.split(sep=".")
    return reduce(getattr, names, module)


def computeStatistics(model, dataloader, modules):
    # Register hooks
    for nam, mod in model.named_modules():
        print(nam)
        if nam in modules:
            module = get_module_by_name(model, nam)
            module.register_forward_hook(getLayerName(nam))

    # Create custom batch
    batch = {"image": [], "label": []}
    for image, label in dataloader:
        batch["image"].append(image)
        batch["label"].append(label)

    batch["image"] = torch.stack(batch["image"])
    batch["image"] = torch.squeeze(batch["image"])
    batch["image"] = batch["image"].to(device)
    model(batch["image"])


def getLayerName(name):
    # Layer shape is CONV so it should be B, C, H, W
    def statisticHook(model, input_data, output):
        input_data = input_data[0]
        # Do we need this?

        output_shape = list(output.shape)

        # Select every axis except the first one that includes batches
        output_dim_axis = [i for i in range(len(output_shape))][1:]

        # Area of the filters
        output_area = math.prod(output_shape[1:])

        # Average over all dimensions except batch
        output_averages = torch.divide(
            torch.count_nonzero(torch.greater_equal(output, 0), dim=output_dim_axis), output_area
        )

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
        w, lamb = torch.linalg.eig(covariance)
        ## Complex sort does not work.
        w, indices = torch.sort(w, dim=0, descending=True)  # Complex sort

        # Plot norm of eigenvalues
        norms, indices = torch.sort(torch.norm(w, dim=1), descending=True)  # Norm sort
        norms = norms.to("cpu").detach().numpy()
        plt.plot(range(0, len(norms)), norms, "xb-")
        plt.savefig(os.path.join(data_root, name + ".jpg"))

        # JSON
        json_data[name] = {
            "avg_nonpositive": output_nonpositive_avg.to("cpu").detach().numpy().tolist(),
            "avg": output_mean_tensor.to("cpu").detach().numpy().tolist(),
            "eigvals": w.to("cpu").detach().numpy().tolist(),
        }
        json_object = json.dumps(json_data, indent=4)
        with open("statistics.json", "w") as outfile:
            outfile.write(json_object)

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


def trainLoop(dataloader, lossFunction, model, loadCheckpoint, WandB, pathOrigin):
    """
    Trains a deep learning model with a variant of gradient decent

    Parameters
    ----------
    dataloader: torch dataloader object
    lossFunction: torch.nn function
    model: torch.nn object
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
            project=train_conf["model_name"],
            # track train_conf and run metadata
            config={
                "learning_rate": train_conf["learning_rate"],
                "architecture": train_conf["model_name"],
                "epochs": train_conf["epochs"],
            },
        )

    # optimizer
    if train_conf["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=train_conf["learning_rate"],
            weight_decay=train_conf["weight_decay"],
        )

    # load model
    if loadCheckpoint:
        # get into folder
        model_name = str(train_conf["model_name"])
        loadCheckpoint(model, optimizer, os.path.join(pathOrigin, "models", model_name))

    # start training
    train_count = 0
    model.train()
    runningLoss = 0
    for i in range(train_conf["epochs"]):
        for input_data, labels in dataloader:
            if train_conf["normalize_images"]:
                input_data = torch.nn.functional.normalize(input_data)  # normalize image
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
        model_name = str(train_conf["model_name"])
        saveCheckpoint(model, optimizer, os.path.join(pathOrigin, "models", model_name))

    return


modules = ["conv1", "layer1.0.conv1", "layer1.1.conv1", "layer2.0.conv1"]
computeStatistics(model, dataloader_dict["hook"], modules)


trainLoop(
    dataloader_dict["train"],
    lossFunction=criterion,
    model=model,
    loadCheckpoint=False,
    WandB=False,
    pathOrigin=data_root,
)
