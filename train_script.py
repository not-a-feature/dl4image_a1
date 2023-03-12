import torch
import torch.nn as nn
from torchvision import models
import wandb
import os
from data_loader import get_dataloaders

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


def trainLoop(dataloader, lossFunction, model, params, loadCheckpoint, WandB, pathOrigin):
    """
    Trains a deep learning model with a variant of gradient decent

    Parameters
    ----------
    dataloader: torch dataloader object
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
            project=params["model_name"],
            # track params and run metadata
            config={
                "learning_rate": params["learning_rate"],
                "architecture": params["model_name"],
                "epochs": params["epochs"],
            },
        )

    # optimizer
    if params["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params["learning_rate"],
            weight_decay=params["weight_decay"],
        )

    # load model
    if loadCheckpoint:
        # get into folder
        model_name = str(params["model_name"])
        loadCheckpoint(model, optimizer, os.path.join(pathOrigin, "models", model_name))

    # start training
    train_count = 0
    model.train()
    runningLoss = 0
    for i in range(params["epochs"]):
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
        model_name = str(params["model_name"])
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
