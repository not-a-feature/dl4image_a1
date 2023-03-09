from data_loader import get_dataloaders
from matplotlib import pyplot as plt

dataloader_train, dataloader_test, dataloader_val = get_dataloaders()

data = dataloader_train.get_batch(32, 7)
