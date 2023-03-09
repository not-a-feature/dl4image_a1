classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# data_root = "/home/jules/Bioinformatik/2.OSLO/Deep_Learning/dl4image_a1"
data_root = "/itf-fi-ml/home/julesk/dl4image_a1"
data_folder = "mandatory1_data"

dataloader_conf = {
    "take_every_nth": 1,
    "random_seed": 12345678,
    "n_test": 3000,
    "n_val": 2000,
}

train_conf = {
    "learningRate": 0.001,
    "weightDecay": 0.01,
    "modelName": "resnet18_6class",
    "epochs": 20,
    "optimizer": "Adam",
    "batch_size": 256,
    "device": "cuda:4",
}
