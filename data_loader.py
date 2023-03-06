import os
from sklearn.model_selection import train_test_split
from collections import Counter

data_root = "/home/jules/Bioinformatik/2.OSLO/Deep_Learning/mandatory_1/mandatory1_data"
classes = ["buildings", "forest", "glacier", "mountain", "sea", "street"]
random_seed = 12345678


def get_splits(n_test=3000, n_val=2000):
    fns = []
    labels = []

    for c in classes:
        class_fn = os.listdir(os.path.join(data_root, c))
        class_fn = [os.path.join(data_root, c, fn) for fn in class_fn]

        class_size = len(class_fn)

        fns.extend(class_fn)
        labels.extend([c] * class_size)

    total_size = len(fns)
    # Size of test and val set in percent
    p_test_val = (n_test + n_val) / total_size

    fn_train, fn_test_val, label_train, label_test_val = train_test_split(
        fns,
        labels,
        test_size=p_test_val,
        random_state=random_seed,
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
        random_state=random_seed,
        shuffle=True,
    )

    # Check if test / train / val set are disjoint
    set_fn_train = set(fn_train)
    set_fn_test = set(fn_test)
    set_fn_val = set(fn_val)

    assert set_fn_test.intersection(set_fn_train) == set()
    assert set_fn_test.intersection(set_fn_val) == set()
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


def load_data():
    pass
