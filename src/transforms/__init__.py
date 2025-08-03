from src.transforms.normalize import Normalize
from src.transforms.scale import RandomScale1D
from src.transforms.stft import STFTTransform, MelSpectrogramTransform, LogTransform
from src.transforms.lfcc import LFCCTransform
from src.transforms.augmentations import (
    AddNoise,
    TimeStretch,
    TimeMasking,
    ComposeAugmentations,
    get_anti_spoofing_augmentations
)