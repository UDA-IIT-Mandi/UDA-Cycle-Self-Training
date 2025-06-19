import os
from torchvision.datasets import SVHN as TorchSVHN
from torchvision import transforms

class SVHN(TorchSVHN):
    """SVHN Dataset Wrapper for domain adaptation compatibility."""
    domains = ["svhn"]
    num_classes = 10

    def __init__(self, root, task=None, split='train', transform=None, download=False):
        # SVHN splits: train, test, extra
        if split not in ['train', 'test', 'extra']:
            raise ValueError("split must be 'train', 'test', or 'extra'")
        data_root = os.path.join(root, "SVHN")
        super().__init__(root=data_root, split=split, download=download, transform=transform)
        self.task = task or "svhn"
        self.classes = [str(i) for i in range(10)]

    @property
    def num_classes(self):
        return 10