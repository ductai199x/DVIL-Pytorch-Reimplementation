import os
import random

import h5py
import pandas as pd

import torch
from torch.utils.data import Dataset
from torchvision.transforms.functional import crop

from tqdm.auto import tqdm
from typing import *


class VideoFact2Dataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        metadata: pd.DataFrame,
        split: Union[None, str] = None,
        only_manipulated: bool = False,
        return_type: Literal["video", "frame"] = "video", 
        crfs: Union[None, List[str]] = None,
        max_num_return_frames: int = 5,
        return_frame_size: Union[None, Tuple[int, int]] = (1080, 1920),
    ):
        self.root_dir = root_dir
        self.split = split
        self.only_manipulated = only_manipulated
        self.metadata = metadata
        if self.split is not None:
            self.metadata = self.metadata.query(f"split == '{self.split}'")
        self.return_type = return_type
        self.crfs = crfs
        self.max_num_return_frames = max_num_return_frames
        self.return_frame_size = return_frame_size

        assert (
            self.metadata["num_frames"].min() >= self.max_num_return_frames
        ), f"num_frames must be <= {self.metadata['num_frames'].min()}"

        self.data_samples = self.create_frame_level_data() if self.return_type == "frame" else self.create_video_level_data()

    def _pad(self, x):
        if self.return_frame_size is not None:
            return crop(x, 0, 0, *self.return_frame_size)
        else:
            return x

    def __len__(self):
        return len(self.data_samples)

    def create_frame_level_data(self):
        samples = []
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc=f"Loading {self.split} frame-level data"):
            if not self.only_manipulated and os.path.exists(f"{self.root_dir}/{row['vid_id']}/orig.hdf5"):
                row_samples = list(zip([f"{self.root_dir}/{row['vid_id']}/orig.hdf5"] * row["num_frames"], range(row["num_frames"])))
                samples.extend(row_samples)
            if os.path.exists(f"{self.root_dir}/{row['vid_id']}/manip.hdf5"):
                row_samples = list(zip([f"{self.root_dir}/{row['vid_id']}/manip.hdf5"] * row["num_frames"], range(row["num_frames"])))
                samples.extend(row_samples)
        return samples

    def create_video_level_data(self):
        samples = []
        for _, row in tqdm(self.metadata.iterrows(), total=len(self.metadata), desc=f"Loading {self.split} video-level data"):
            if not self.only_manipulated and os.path.exists(f"{self.root_dir}/{row['vid_id']}/orig.hdf5"):
                samples.append((f"{self.root_dir}/{row['vid_id']}/orig.hdf5", row["num_frames"]))
            if os.path.exists(f"{self.root_dir}/{row['vid_id']}/manip.hdf5"):
                samples.append((f"{self.root_dir}/{row['vid_id']}/manip.hdf5", row["num_frames"]))
        return samples

    def get_list_frame_idxs(self, vid_length: int) -> List[int]:
        if self.max_num_return_frames == -1:
            return list(range(vid_length))
        else:
            start_idx = random.randint(0, vid_length - self.max_num_return_frames)
            return list(range(start_idx, start_idx + self.max_num_return_frames))
    
    def get_video_sample(self, file_path: str, vid_length: int):
        with h5py.File(file_path, "r") as f:
            if self.crfs is None:
                crf = random.choice(["crf0", "crf23", "crf40"])
            else:
                crf = random.choice(self.crfs)

            idxs = self.get_list_frame_idxs(vid_length)
            frames = torch.from_numpy(f[crf][idxs])
            if "mask" in f:
                masks = torch.from_numpy(f["mask"][idxs])
                masks = masks / masks.max()
                if "inpainting" in file_path:
                    masks = 1 - masks
                label = 1  # manipulated
            else:
                masks = torch.zeros((frames.shape[0], *frames.shape[-2:])).int()
                label = 0

            initial_shape = torch.tensor(frames.shape)[-2:] # (H, W)
            frames = self._pad(frames).float() / 255.0
            masks = self._pad(masks)
            
            return frames, label, masks, initial_shape
        
    def get_frame_sample(self, file_path: str, frame_idx: int):
        with h5py.File(file_path, "r") as f:
            if self.crfs is None:
                crf = random.choice(["crf0", "crf23", "crf40"])
            else:
                crf = random.choice(self.crfs)

            frame = torch.from_numpy(f[crf][frame_idx])
            if "mask" in f:
                mask = torch.from_numpy(f["mask"][frame_idx])
                mask = mask / mask.max()
                if "inpainting" in file_path:
                    mask = 1 - mask
                label = 1
            else:
                mask = torch.zeros(frame.shape[-2:]).int()
                label = 0

            initial_shape = torch.tensor(frame.shape)[-2:] # (H, W)
            frame = self._pad(frame).float() / 255.0
            mask = self._pad(mask)

            return frame, label, mask, initial_shape

    def __getitem__(self, idx):
        if self.return_type == "frame":
            return self.get_frame_sample(*self.data_samples[idx])
        else:
            return self.get_video_sample(*self.data_samples[idx])
        