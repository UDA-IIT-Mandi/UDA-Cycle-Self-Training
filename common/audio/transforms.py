import torch
import numpy as np

class WeakAugment(object):
    def __call__(self, waveform):
        """Apply weak augmentation to waveform."""
        if np.random.random() < 0.5:
            shift = int(np.random.uniform(-0.1, 0.1) * len(waveform))
            waveform = torch.roll(waveform, shift, dims=0)
        return waveform

class StrongAugment(object):
    def __call__(self, waveform):
        """Apply strong augmentation to waveform."""
        shift = int(np.random.uniform(-0.2, 0.2) * len(waveform))
        waveform = torch.roll(waveform, shift, dims=0)
        
        waveform = waveform + torch.randn_like(waveform) * 0.01
        
        return waveform

class TransformFixMatch(object):
    def __init__(self):
        self.weak = WeakAugment()
        self.strong = StrongAugment()
        
    def __call__(self, x):
        weak = self.weak(x)
        strong = self.strong(x)
        return weak, strong

__all__ = ['WeakAugment', 'StrongAugment', 'TransformFixMatch']
