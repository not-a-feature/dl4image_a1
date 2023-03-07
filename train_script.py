import torch
import torch.nn as nn
from torchvision import models
import wandb
import os
from data_loader import get_dataloaders

from config import train_conf as params
from config import classes

# specify pathOrigin here
pathOrigin = "/home/jules/Bioinformatik/2.OSLO/Deep_Learning/dl4image_a1"

# example model resnet, maybe add a layer to match our image sizes in the beginning
model = models.resnet18(weights="DEFAULT")
model.fc = nn.Linear(512, len(classes))
model = model.float()


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


dataloader_train, dataloader_test, dataloader_val = get_dataloaders()

#################


def trainLoop(dataLoader, lossFunction, model, hyperparameters, loadCheckpoint, WandB, pathOrigin):
    """
    Trains a deep learning model with a variant of gradient decent

    Parameters
    ----------
    dataLoader: torch dataLoader object
    lossFunction: torch.nn function
    model: torch.nn object
    hyperparameters: dict
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
            project=hyperparameters["modelName"],
            # track hyperparameters and run metadata
            config={
                "learning_rate": hyperparameters["learningRate"],
                "architecture": hyperparameters["modelName"],
                "epochs": hyperparameters["epochs"],
            },
        )

    # optimizer
    if hyperparameters["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=hyperparameters["learningRate"],
            weight_decay=hyperparameters["weightDecay"],
        )

    # load model
    if loadCheckpoint:
        # get into folder
        os.chdir(pathOrigin + "/models")
        loadCheckpoint(model, optimizer, pathOrigin + "/models/" + hyperparameters["modelName"])

    # start training
    train_count = 0
    model.train()
    runningLoss = 0
    for i in range(hyperparameters["epochs"]):
        batch_idx_range = range(len(dataLoader) // params["batch_size"])
        for batch_idx in batch_idx_range:
            sample = dataLoader.get_batch(params["batch_size"], batch_idx)

            input_data, labels = sample["image"], sample["label"]

            input_data.to(device)
            labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # calculate loss + backprop
            pred = model(input_data)
            loss = lossFunction(pred, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.detach().cpu().item()
            train_count += 1
            print("Epoch: ", i, "current loss: ", runningLoss / train_count)

            ### implement early stopping here #####

            ########################################

            # save checkpoint at each end of epoch
        saveCheckpoint(model, optimizer, pathOrigin + "/models/" + hyperparameters["modelName"])

    return


device = "cpu"
criterion = torch.nn.CrossEntropyLoss()

trainLoop(
    dataloader_train,
    lossFunction=criterion,
    model=model,
    hyperparameters=params,
    loadCheckpoint=False,
    WandB=False,
    pathOrigin=pathOrigin,
)
