classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

# data_root = "/home/user3574/PycharmProjects/dl4image_a1"
data_root = "/itf-fi-ml/home/julesk/dl4image_a1"
data_folder = "mandatory1_data"

dataloader_conf = {
    "take_every_nth": 10,
    "random_seed": 12345678,
    "n_test": 3000,
    "n_val": 2000,
    "included_flipped": True,
    "num_workers": 4,
    "batch_size": 512,
}

train_conf = {
    "model_name": "resnet18_6class",
    "learning_rate": 0.001,
    "weight_decay": 0.01,
    "epochs": 20,
    "normalize_images": True,
    "optimizer": "Adam",
    "device": "cuda:4",
    # "device": "cpu",
    "input_size": 150,
}
