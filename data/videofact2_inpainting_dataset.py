import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize

from .videofact2_dataset import VideoFact2Dataset
from typing import *


class VideoFact2InpaintingDataset(Dataset):
    def __init__(
        self,
        videofact2_dataset_obj: VideoFact2Dataset,
        transforms: Union[torch.nn.Module, None] = None,
        resize: Union[Tuple[int, int], None] = None,
    ):
        self.dataset = videofact2_dataset_obj
        self.transforms = transforms if transforms is not None else Compose([])
        self.resize = Resize(resize, antialias=True) if resize is not None else Compose([])

        assert self.dataset.return_frame_size is None, "return_frame_size must be None"
        assert self.dataset.return_type == "video", "return_type must be 'video'"

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        frames, label, masks, shape = sample

        return (
            self.transforms(self.resize(frames)),
            (self.resize(masks) > 0.5).int(),
            label,
        )
