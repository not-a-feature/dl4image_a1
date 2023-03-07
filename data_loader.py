import os
from sklearn.model_selection import train_test_split
from collections import Counter
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader


data_root = "/home/jules/Bioinformatik/2.OSLO/Deep_Learning/dl4image_a1/mandatory1_data"
classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

conf = {
    "random_seed": 12345678,
    "n_test": 3000,
    "n_val": 2000,
    # "batch_size": 8,
}


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
        label_train: list[str], list of labels of train set.
        label_test: list[str], list of labels of test set.
        label_val: list[str], list of labels of validation set
    """
    fns = []
    labels = []

    for c in classes:
        class_fn = os.listdir(os.path.join(data_root, c))
        class_fn = [os.path.join(data_root, c, fn) for fn in class_fn]

        class_size = len(class_fn)

        fns.extend(class_fn)
        labels.extend([c] * class_size)

    total_size = len(fns)

    if total_size <= (n_test + n_val):
        raise RuntimeError("Test / Val set is too large.")

    # Size of test and val set in percent
    p_test_val = (n_test + n_val) / total_size

    fn_train, fn_test_val, label_train, label_test_val = train_test_split(
        fns,
        labels,
        test_size=p_test_val,
        random_state=conf["random_seed"],
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
        random_state=conf["random_seed"],
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

    print("Name    c   |", "N".rjust(pad), "| Composition")
    print("----------------------------------------------")
    print("Train Split |", str(len(fn_train)).rjust(pad), "|", Counter(label_train))
    print("Test Split  |", str(len(fn_test)).rjust(pad), "|", Counter(label_test))
    print("Val Split   |", str(len(fn_val)).rjust(pad), "|", Counter(label_val))

    return fn_train, fn_test, fn_val, label_train, label_test, label_val


def load_image(path):
    """Loads an image to a numpy array.

    Parameters
    ----------
        path: str, path to image

    Returns
    -------
        img: np.array, image
    """

    return np.asarray(Image.open(path))


def load_data():
    """
    Loads image data and returns train, test and validation splits.

    Parameters
    ----------
        None

    Returns
    ----------
        data_train: list[np.array(n, n, 3)], list of images of train set.
        data_test: list[np.array(n, n, 3)], list of images of test set.
        data_val: list[np.array(n, n, 3)], list of images of val set.
        label_train: list[str], list of labels of train set.
        label_test: list[str], list of labels of test set.
        label_val: list[str], list of labels of validation set
    """

    fn_train, fn_test, fn_val, label_train, label_test, label_val = get_splits(
        n_test=conf["n_test"], n_val=conf["n_val"]
    )

    data_train = [load_image(fn) for fn in fn_train]
    data_test = [load_image(fn) for fn in fn_test]
    data_val = [load_image(fn) for fn in fn_val]

    return data_train, data_test, data_val, label_train, label_test, label_val


load_data()
