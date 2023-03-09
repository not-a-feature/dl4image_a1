import os
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from config import dataloader_conf, classes, data_root, data_folder
from torch import is_tensor, from_numpy, permute, stack
from torch import nn
from torchvision import transforms
from skimage import io


class Landmarks(Dataset):
    """Landmarks dataset."""

    def __init__(self, img_path, labels, transform):
        """
        Args:
            root_dir (string): Directory with all the images.
            # todo fix
        """
        self.img_path = img_path
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.img_path[idx])
        image = from_numpy(image)
        image = permute(image, (2, 0, 1))
        # image = image[None, :]
        image = image.float()

        label = np.zeros(len(classes))
        label[self.labels[idx]] = 1
        label = from_numpy(label)
        # label = label[None, :]

        # sample = {"image": image, "label": label}

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_batch(self, batch_size, idx):
        batch = {"image": [], "label": []}

        for i in range(batch_size):
            image, label = self.__getitem__(idx * batch_size + i)
            batch["image"].append(image)
            batch["label"].append(label)

        batch["image"] = stack(batch["image"])
        batch["label"] = stack(batch["label"])
        return batch


def get_splits(n_test=3000, n_val=2000):
    """
    Gets all file-names and splits them into train, test and val sets.

    Parameters
    ----------
        n_test: int, number of images to include in the test set.
        n_val: int, number of images to include in the validation set.

    Returns
    ----------
        fn_train:, list[str], list of image-paths of train set.
        fn_test: list[str], list of image-paths of test set.
        fn_val: list[str], list of image-paths of validation set.
        label_train: list[int], list of labels of train set.
        label_test: list[int], list of labels of test set.
        label_val: list[int], list of labels of validation set
    """
    fns = []
    labels = []

    for i, c in enumerate(classes):
        class_fn = os.listdir(os.path.join(data_root, data_folder, c))
        class_fn = [os.path.join(data_root, data_folder, c, fn) for fn in class_fn]

        class_size = len(class_fn)

        fns.extend(class_fn)
        labels.extend([i] * class_size)

    fns = fns[:: dataloader_conf["take_every_nth"]]
    labels = labels[:: dataloader_conf["take_every_nth"]]

    n_test = n_test // dataloader_conf["take_every_nth"]
    n_val = n_val // dataloader_conf["take_every_nth"]

    total_size = len(fns)

    if total_size <= (n_test + n_val):
        raise RuntimeError("Test / Val set is too large.")

    # Size of test and val set in percent
    p_test_val = (n_test + n_val) / total_size

    fn_train, fn_test_val, label_train, label_test_val = train_test_split(
        fns,
        labels,
        test_size=p_test_val,
        random_state=dataloader_conf["random_seed"],
        shuffle=True,
    )

    # Size of test set in percent
    p_test = n_test / (n_test + n_val)

    # Size of val set in percent
    p_val = n_val / (n_test + n_val)

    fn_test, fn_val, label_test, label_val = train_test_split(
        fn_test_val,
        label_test_val,
        test_size=p_val,  # Important: we split the (test-val) set into test and val. ´
        random_state=dataloader_conf["random_seed"],
        shuffle=True,
    )

    # Check if test / train / val set are disjoint
    set_fn_train = set(fn_train)
    set_fn_test = set(fn_test)
    set_fn_val = set(fn_val)

    assert set_fn_train.intersection(set_fn_test) == set()
    assert set_fn_train.intersection(set_fn_val) == set()
    assert set_fn_test.intersection(set_fn_val) == set()

    # Check that length of data / label is equal
    assert len(set_fn_train) == len(set_fn_train)
    assert len(set_fn_test) == len(set_fn_test)
    assert len(set_fn_val) == len(set_fn_val)

    pad = len(str(total_size))

    print("Name        |", "N".rjust(pad), "| Composition")
    print("----------------------------------------------")
    print("Train Split |", str(len(fn_train)).rjust(pad), "|", Counter(label_train))
    print("Test Split  |", str(len(fn_test)).rjust(pad), "|", Counter(label_test))
    print("Val Split   |", str(len(fn_val)).rjust(pad), "|", Counter(label_val))

    return fn_train, fn_test, fn_val, label_train, label_test, label_val


transform = nn.Sequential(transforms.Resize((150, 150)))


def get_dataloaders():
    """
    Returns 3 dataloaders.
    """
    data_train, data_test, data_val, label_train, label_test, label_val = get_splits()

    dataloader_train = Landmarks(data_train, label_train, transform)
    dataloader_test = Landmarks(data_test, label_test, transform)
    dataloader_val = Landmarks(data_val, label_val, transform)

    return {"train": dataloader_train, "test": dataloader_test, "val": dataloader_val}
