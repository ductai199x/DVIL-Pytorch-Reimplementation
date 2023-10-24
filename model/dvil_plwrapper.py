from lightning.pytorch.utilities.types import OptimizerLRScheduler
import wandb
from wandb.sdk.wandb_run import Run

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchvision.models.optical_flow.raft import Raft_Small_Weights, raft_small
from torchvision.transforms import Normalize

from lightning.pytorch import LightningModule
from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex
from torchmetrics.functional.classification import dice

from .dvil import DVIL_PyTorch
from typing import *


class DVIL_PLWrapper(LightningModule):
    def __init__(
        self,
        model_config,
        training_config,
    ):
        super().__init__()

        self.model = DVIL_PyTorch(**model_config)
        self.training_config = training_config

        self.train_f1 = BinaryF1Score()
        self.train_jaccard = BinaryJaccardIndex()
        self.val_f1 = BinaryF1Score()
        self.val_jaccard = BinaryJaccardIndex()

        self.save_hyperparameters()

        self.raft_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.raft = raft_small(weights=Raft_Small_Weights.DEFAULT).to(self.raft_device)
        self.raft_transforms = Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        self.raft = self.raft.eval()

    def forward(self, x, aug_x):
        return self.model(x, aug_x)

    @torch.no_grad()
    def log_loc_output(self, x, gt_mask, pred_mask, step_idx):
        x = x.detach().cpu()
        gt_mask = gt_mask.float().detach().cpu()
        pred_mask = pred_mask.float().detach().cpu()
        logger = self.logger.experiment
        if isinstance(logger, Run):
            log_images = []
            log_images.append(wandb.Image(x, caption="input"))
            log_images.append(wandb.Image(gt_mask, caption="gt_mask"))
            log_images.append(wandb.Image(pred_mask, caption="pred_mask"))
            logger.log({"train_loc_output": log_images}, step=step_idx)
        elif isinstance(logger, SummaryWriter):
            logger.add_images("train_loc_x", x, dataformats="CHW", global_step=step_idx)
            logger.add_images("train_loc_gt", gt_mask, dataformats="HW", global_step=step_idx)
            logger.add_images("train_loc_pred", pred_mask, dataformats="HW", global_step=step_idx)
        else:
            pass

    @torch.no_grad()
    def get_flows(self, frames):
        T, C, H, W = frames.shape

        forward_flow_t_idx = torch.arange(1, T - 1)
        backward_flow_t_idx = torch.arange(T - 2, 0, -1)
        forward_flow_t_next_idx = forward_flow_t_idx + 1
        backward_flow_t_prev_idx = backward_flow_t_idx - 1

        flow_left_side = torch.cat([forward_flow_t_idx, backward_flow_t_idx])
        flow_right_side = torch.cat([forward_flow_t_next_idx, backward_flow_t_prev_idx])

        left_side = self.raft_transforms(frames[flow_left_side]).to(self.raft_device)
        right_side = self.raft_transforms(frames[flow_right_side]).to(self.raft_device)

        flows = (
            self.raft(left_side.contiguous(), right_side.contiguous(), num_flow_updates=12)[-1].detach()
        )
        return flows

    def training_step(self, batch, batch_idx):
        frames_batches, gt_masks_batches, _ = batch

        loss = 0
        for frames, gt_masks in zip(frames_batches, gt_masks_batches):
            num_frames = frames.shape[0]
            # Pad the batch with the first and last frames
            idxs = torch.arange(-1, num_frames + 1)
            idxs[0] = 0
            idxs[-1] = num_frames - 1
            frames_ = frames[idxs]
            flows = self.get_flows(frames_)

            pred_masks = self(frames_, flows).to(self.device)
            loss += F.binary_cross_entropy(pred_masks, gt_masks.float())
            # loss += 1 - dice(pred_masks, gt_masks).requires_grad_(True)
            

            self.train_f1(pred_masks, gt_masks)
            self.train_jaccard(pred_masks, gt_masks)

        # loss += torch.norm(self.model.intra_frame_residual.hpf.hpf.weight, p=2) * 1e-2

        if self.global_step % 200 == 0:
            self.log_loc_output(frames[0], gt_masks[0], pred_masks[0], self.global_step)

        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        self.log("train_f1", self.train_f1, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "train_jaccard", self.train_jaccard, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        frames_batches, gt_masks_batches, _ = batch

        loss = 0
        for frames, gt_masks in zip(frames_batches, gt_masks_batches):
            num_frames = frames.shape[0]
            # Pad the batch with the first and last frames
            idxs = torch.arange(-1, num_frames + 1)
            idxs[0] = 0
            idxs[-1] = num_frames - 1
            frames_ = frames[idxs]
            flows = self.get_flows(frames_)

            pred_masks = self(frames_, flows).to(self.device)
            loss += F.binary_cross_entropy(pred_masks, gt_masks.float())
            # loss += 1 - dice(pred_masks, gt_masks)

            self.val_f1(pred_masks, gt_masks)
            self.val_jaccard(pred_masks, gt_masks)

        if self.global_step % 200 == 0:
            self.log_loc_output(frames[0], gt_masks[0], pred_masks[0], self.global_step)

        self.log("val_loss", loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log(
            "val_jaccard", self.val_jaccard, on_step=False, on_epoch=True, prog_bar=False, sync_dist=True
        )

    def configure_optimizers(self):
        # optimizer = torch.optim.Adam(
        #     self.parameters(),
        #     lr=self.training_config["lr"],
        #     weight_decay=self.training_config["weight_decay"],
        # )
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.training_config["lr"],
            momentum=0.95,
        )
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=self.training_config["decay_step"], gamma=self.training_config["decay_rate"]
        )
        return [optimizer], [scheduler]
