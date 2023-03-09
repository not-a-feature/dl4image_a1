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

# Get Dataloaders
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
