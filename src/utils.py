import numpy as np
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    RandFlip,
    RandZoom,
    RandRotate,
    NormalizeIntensity,
)

def get_transforms(mean=12, std=24):
    train_transforms = Compose([
        EnsureChannelFirst(),
        NormalizeIntensity(subtrahend=mean, divisor=std),
        RandFlip(spatial_axis=1, prob=0.7),
        RandRotate(
            mode=("bilinear"),
            range_x=np.pi / 18,
            range_y=np.pi / 18,
            range_z=np.pi / 18,
            prob=0.7,
            padding_mode=("reflection")
        ),
        RandZoom(min_zoom=0.9, max_zoom=1.2, prob=0.5),
    ])

    val_transforms = Compose([
        EnsureChannelFirst(),
        NormalizeIntensity(subtrahend=mean, divisor=std)
    ])
    
    return train_transforms, val_transforms