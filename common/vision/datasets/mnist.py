import os
from torchvision.datasets import MNIST as TorchMNIST
from torchvision import transforms

class MNIST(TorchMNIST):
    """MNIST Dataset Wrapper for domain adaptation compatibility."""
    domains = ["mnist"]
    num_classes = 10

    def __init__(self, root, task=None, split='train', transform=None, download=False):
        # MNIST splits: train, test
        if split not in ['train', 'test']:
            raise ValueError("split must be 'train' or 'test'")
        data_root = os.path.join(root, "MNIST")
        train = split == 'train'
        super().__init__(root=data_root, train=train, download=download, transform=transform)
        self.task = task or "mnist"
        self.classes = [str(i) for i in range(10)]

    @property
    def num_classes(self):
        return 10