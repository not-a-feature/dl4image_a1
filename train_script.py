import torch
from torchvision import models
import torch.optim as optim
import wandb
import os

# specify pathOrigin here
#pathOrigin = "..."

# example model resnet, maybe add a layer to match our image sizes in the beginning
model = models.resnet18(pretrained=True)
device = "cpu"
params = {"learningRate": 0.001,
                   "weightDecay": 0.01,
                   "modelName": "resnet18",
                   "epochs": 20,
                    "optimizer": "Adam"}
criterion = torch.nn.CrossEntropyLoss()



def saveCheckpoint(model, optimizer, filename):
    """
    saves current model and optimizer step

    model: nn.model
    optimizer: torch.optim.optimzer class
    filename: string
    """
    checkpoint = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict()}
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
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    print("checkpoint loaded")
    return

##### dataLoader






#################


def trainLoop(dataLoader, lossFunction, model, hyperparameters, optimizer, loadCheckpoint, WandB, pathOrigin):
    """
    Trains a deep learning model with a variant of gradient decent

    Parameters
    ----------
    dataLoader: torch dataLoader object
    lossFunction: torch.nn function
    model: torch.nn object
    hyperparameters: dict
    optimizer: torch.optim object
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
            }
        )

    # optimizer
    if hyperparameters["optimizer"] == "Adam":
        optimizer = torch.Adam(model.parameters(),
                               lr = hyperparameters["learningRate"],
                               weight_decay = hyperparameters["weightDecay"])


    # load model
    if loadCheckpoint:
        # get into folder
        os.chdir(pathOrigin + "/models")
        loadCheckpoint(model, optimizer, pathOrigin + "/models/" + hyperparameters["modelName"])

    # start training
    model.train()
    runningLoss = 0
    for i in range(hyperparameters["epochs"]):
        for input, labels in dataLoader:
            input.to(device)
            labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # calculate loss + backprop
            pred = model(input)
            loss = lossFunction(pred, labels)
            loss.backward()
            optimizer.step()

            runningLoss += loss.detach().cpu().item()

            print("Epoch: ", i, "current loss: ", runningLoss)

            ### implement early stopping here #####



            ########################################

            # save checkpoint at each end of epoch
            saveCheckpoint(model, optimizer, pathOrigin + "/models/" + hyperparameters["modelName"])
            
    return




