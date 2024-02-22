import numpy as np
import torchvision.datasets as datasets

# Taken from https://github.com/pluskid/fitting-random-labels/blob/master/cifar10_data.py


class CIFARRandomLabelsBase:
    """CIFAR10 dataset, with support for randomly corrupt labels.
    Params
    ------
    num_classes: int
      The number of classes in the dataset.
    corrupt_prob: float
      Default 0.0. The probability of a label being replaced with
      random label.
    """

    def __init__(self, num_classes, corrupt_prob=0.0, **kwargs):
        super(CIFARRandomLabelsBase, self).__init__(**kwargs)
        self.n_classes = num_classes
        if corrupt_prob > 0:
            self.corrupt_labels(corrupt_prob)

    def corrupt_labels(self, corrupt_prob):
        labels = np.array(self.targets)
        np.random.seed(12345)
        mask = np.random.rand(len(labels)) <= corrupt_prob
        rnd_labels = np.random.choice(self.n_classes, mask.sum())
        labels[mask] = rnd_labels
        # we need to explicitly cast the labels from npy.int64 to
        # builtin int type, otherwise pytorch will fail...
        labels = [int(x) for x in labels]

        self.targets = labels


def get_random_cifar_dataset(dataset, num_classes, corrupt_prob=0.0, **kwargs):

    class CIFARRandomLabels(CIFARRandomLabelsBase, dataset):
        pass

    return CIFARRandomLabels(num_classes, corrupt_prob, **kwargs)
