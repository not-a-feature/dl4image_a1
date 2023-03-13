import os
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from torch import is_tensor, from_numpy, permute, stack, nn
from torchvision import transforms

from skimage import io

from config import *


class Landmarks(Dataset):
    """Landmarks dataset."""

    def __init__(self, img_path, labels, transform=False):
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
        """
        batch = {"image": [], "label": []}

        for i in range(batch_size):
            image, label = self.__getitem__(idx * batch_size + i)
            batch["image"].append(image)
            batch["label"].append(label)

        batch["image"] = stack(batch["image"])
        batch["label"] = stack(batch["label"])
        return batch
        """
        if is_tensor(idx):
            idx = idx.tolist()

        image = io.imread(self.img_path[idx])
        image = from_numpy(image)
        image = permute(image, (2, 0, 1))
        image = image.float()

        label = np.zeros(len(classes))
        label[self.labels[idx]] = 1
        label = from_numpy(label)

        if self.transform:
            image = self.transform(image)

        return image, label


def get_splits(n_test=3000, n_val=2000, num_hook=200, include_augmented=True):
    """
    Gets all file-names and splits them into train, test and val sets.

    Parameters
    ----------
        n_test: int, number of images to include in the test set.
        n_val: int, number of images to include in the validation set.
        num_hook: int, number of images to use for the hook.
        include_augmented: bool, include flipped images.

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

        class_fn = [fn for fn in class_fn if include_augmented or "flipped" not in fn]
        class_fn = [os.path.join(data_root, data_folder, c, fn) for fn in class_fn]

        class_size = len(class_fn)

        fns.extend(class_fn)
        labels.extend([i] * class_size)

    fns = fns[:: dataloader_conf["take_every_nth"]]
    labels = labels[:: dataloader_conf["take_every_nth"]]

    n_test = n_test // dataloader_conf["take_every_nth"]
    n_val = n_val // dataloader_conf["take_every_nth"]

    total_size = len(fns)

    # Hook
    fn_hook = fns[:: round(total_size / num_hook)]
    label_hook = labels[:: round(total_size / num_hook)]

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
    print("Test  Split |", str(len(fn_test)).rjust(pad), "|", Counter(label_test))
    print("Val   Split |", str(len(fn_val)).rjust(pad), "|", Counter(label_val))
    print("Hook  Split |", str(len(fn_hook)).rjust(pad), "|", Counter(label_hook))

    out = {
        "fn_train": fn_train,
        "fn_test": fn_test,
        "fn_val": fn_val,
        "fn_hook": fn_hook,
        "label_train": label_train,
        "label_test": label_test,
        "label_val": label_val,
        "label_hook": label_hook,
    }

    return out


def get_dataloaders():
    """
    Returns 3 dataloaders.
    """
    splits = get_splits(
        dataloader_conf["n_test"],
        dataloader_conf["n_val"],
        dataloader_conf["num_hook"],
        dataloader_conf["included_flipped"],
    )

    transform = nn.Sequential(transforms.Resize((150, 150)))

    landmarks_train = Landmarks(splits["fn_train"], splits["label_train"], transform)
    landmarks_test = Landmarks(splits["fn_test"], splits["label_test"], transform)
    landmarks_val = Landmarks(splits["fn_val"], splits["label_val"], transform)
    landmarks_hook = Landmarks(splits["fn_hook"], splits["label_hook"], transform)

    dataloader_train = DataLoader(
        landmarks_train,
        batch_size=dataloader_conf["batch_size"],
        num_workers=dataloader_conf["num_workers"],
    )
    dataloader_test = DataLoader(
        landmarks_test,
        batch_size=dataloader_conf["batch_size"],
        num_workers=dataloader_conf["num_workers"],
    )
    dataloader_val = DataLoader(
        landmarks_val,
        batch_size=dataloader_conf["batch_size"],
        num_workers=dataloader_conf["num_workers"],
    )

    dataloader_hook = DataLoader(
        landmarks_hook,
        batch_size=1,
        num_workers=1,
    )

    out = {
        "train": dataloader_train,
        "test": dataloader_test,
        "val": dataloader_val,
        "hook": dataloader_hook,
    }
    return out
