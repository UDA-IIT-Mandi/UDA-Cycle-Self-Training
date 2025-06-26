import os
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

class DCASE(Dataset):
    """
    DCASE TAU Urban Acoustic Scenes 2020 Mobile dataset.
    Args:
        root (string): Root directory of dataset where directory ``DCASE`` exists.
        task (string): One of 'device_a', 'device_b', 'device_c', etc.
        split (string, optional): The dataset split, supports ``train``, ``test``
        transform (callable, optional): A function/transform that takes in a waveform and returns a transformed version.
        target_transform (callable, optional): A function/transform that takes in the target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and puts it in root directory.
            If dataset is already downloaded, it is not downloaded again.
    """

    def __init__(self, root, task, split='train', transform=None, target_transform=None, download=False):
        super(DCASE, self).__init__()
        self.root = os.path.expanduser(root)
        self.task = task
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        if download:
            pass

        self.target_sr = 32000
        self.max_length = 10 * self.target_sr  # 10 seconds

        if split == 'train':
            if task == 'source':
                data_path = os.path.join(self.root, 'train', 'source')
            else:
                data_path = os.path.join(self.root, 'train', 'target')
        elif split == 'test':
            if task == 'source':
                data_path = os.path.join(self.root, 'test', 'source')
            else:
                data_path = os.path.join(self.root, 'test', 'target')
        else:
            raise ValueError(f"Split {split} not recognized. Use 'train' or 'test'")

        self.data, self.labels, self.devices = self._load_data(data_path)
        
        unique_labels = sorted(set(self.labels))
        self.label_to_idx = {label: i for i, label in enumerate(unique_labels)}
        self.classes = unique_labels
        self.num_classes = len(unique_labels)

    def _load_data(self, path):
        """Load audio files and corresponding labels."""
        files = []
        labels = []
        devices = []
        
        if not os.path.exists(path):
            raise RuntimeError(f'Dataset not found at {path}')

        for filename in os.listdir(path):
            if filename.endswith('.wav'):
                # Parse filename: scene-[extra]-device.wav
                parts = filename.replace('.wav', '').split('-')
                scene = parts[0]
                device = parts[-1]
                
                files.append(os.path.join(path, filename))
                labels.append(scene)
                devices.append(device)

        return files, labels, devices

    def _load_audio(self, filepath):
        """Load and preprocess audio file."""
        waveform, sr = librosa.load(filepath, sr=self.target_sr, mono=True)
        
        # Pad or truncate to fixed length
        if len(waveform) < self.max_length:
            pad_length = self.max_length - len(waveform)
            waveform = np.pad(waveform, (0, pad_length))
        else:
            waveform = waveform[:self.max_length]
            
        return torch.tensor(waveform, dtype=torch.float32)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (waveform, target, domain, device) where target is index of the target class.
        """
        filepath = self.data[index]
        waveform = self._load_audio(filepath)
        target = self.label_to_idx[self.labels[index]]
        device = self.devices[index]
        domain = self.task  # Use task as domain identifier

        if self.transform is not None:
            waveform = self.transform(waveform)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return waveform, target, domain, device

    def __len__(self):
        return len(self.data)

__all__ = ['DCASE']