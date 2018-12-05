import math
import numpy as np

from fatd.exceptions import MissingImplementationError

def train_test_split(data, labels, train_share=0.8, stratified=True, seed=None):
    # TODO: Unstratified is unsupported now
    if not stratified:
        message = 'Un-stratified splitter is unsupported yet.'
        raise MissingImplementationError(message)

    if seed is None:
        seed = np.random.randint(0, 4294967295)
    np.random.seed(seed=seed)

    n_data_samples = data.shape[0]
    if n_data_samples < 10:
        # TODO: Fix exception
        raise Exception('Not enough data.')

    # TODO: No labels, for named arrays
    data_indices = list(range(n_data_samples))

    # n_train_samples = math.floor(train_share*n_data_samples)
    # n_test_samples = n_data_samples - n_train_samples

    unique_labels = np.unique(labels)
    n_unique_labels = unique_labels.shape[0]

    # Get a list of indices for each class
    label_indices = {}
    label_count = {}
    for i, v in enumerate(labels):
        if v in label_indices:
            label_indices[v].append(i)
            label_count[v] += 1
        else:
            label_indices[v] = [i]
            label_count[v] = 1

    samples_per_class = {}
    for i in label_count:
        n_train_class_samples = math.floor(train_share*label_count[i])
        n_test_class_samples = label_count[i] - n_train_class_samples
        samples_per_class[i] = (n_train_class_samples, n_test_class_samples)
    # samples_per_class = np.array(samples_per_class)

    train_indices = []
    test_indices = []
    for v in samples_per_class:
        n_train_smaples, n_test_smaples = samples_per_class[v]
        ci = np.array(label_indices[v])

        train_id = np.random.choice(len(ci), size=n_train_smaples, replace=False)
        train_mask = np.array([(i in train_id) for i in range(len(ci))])

        train_indices.append(ci[train_mask])
        test_indices.append(ci[~train_mask])

    train_indices = np.concatenate(train_indices)
    test_indices = np.concatenate(test_indices)

    return train_indices, test_indices
