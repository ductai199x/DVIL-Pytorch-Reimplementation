import os
import rich
import argparse
import yaml
import wandb
import pandas as pd

import torch
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.strategies import DDPStrategy
from typing import *

from data.videofact2_dataset import VideoFact2Dataset
from data.videofact2_inpainting_dataset import VideoFact2InpaintingDataset
from model.dvil_plwrapper import DVIL_PLWrapper

torch.set_float32_matmul_precision("high")

parser = argparse.ArgumentParser()

ds_root_dir = "/media/nas2/videofact2_data"
ds_metadata_path = f"{ds_root_dir}/metadata.csv"


def prepare_model(args: dict[str, Any]) -> DVIL_PLWrapper:
    if args is None:
        raise ValueError("args is None")
    prev_ckpt = args.get("prev_ckpt")
    model_config = args["model_args"]
    training_config = args["training_args"]
    if prev_ckpt:
        print(f"Loading from checkpoint: {prev_ckpt}...")
        return DVIL_PLWrapper.load_from_checkpoint(prev_ckpt)
    else:
        return DVIL_PLWrapper(model_config, training_config)


def prepare_datasets(args: dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    metadata = pd.read_csv(ds_metadata_path)
    inpainting_subset = metadata.query("vid_id.str.contains('inpainting')")
    train_ds = VideoFact2Dataset(
        ds_root_dir,
        inpainting_subset,
        "train",
        only_manipulated=True,
        crfs=["crf0", "crf23"],
        max_num_return_frames=args["training_args"]["max_seq_length"],
        return_frame_size=None,
    )
    train_ds = VideoFact2InpaintingDataset(train_ds, resize=args["model_args"]["img_size"])
    val_ds = VideoFact2Dataset(
        ds_root_dir,
        inpainting_subset,
        "val",
        only_manipulated=True,
        crfs=["crf0", "crf23"],
        max_num_return_frames=args["training_args"]["max_seq_length"],
        return_frame_size=None,
    )
    val_ds = VideoFact2InpaintingDataset(val_ds, resize=args["model_args"]["img_size"])
    train_loader = DataLoader(
        train_ds,
        batch_size=args["training_args"]["batch_size"],
        shuffle=True,
        num_workers=args["training_args"]["num_workers"],
        persistent_workers=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args["training_args"]["batch_size"],
        shuffle=False,
        num_workers=args["training_args"]["num_workers"],
        persistent_workers=True,
    )
    return train_loader, val_loader


def prepare_logger(args: dict[str, Any]) -> Tuple[Optional[Union[WandbLogger, TensorBoardLogger]], str]:
    if args["fast_dev_run"]:
        return None, None

    if args.get("logger") is None:
        logger_method = "tensorboard"
    else:
        logger_method = args["logger"]
    args_log_dir = args["log_dir"]
    args_version = args["version"]
    args_uid = args["uid"]

    if logger_method == "tensorboard":
        logger = TensorBoardLogger(
            save_dir=os.getcwd(),
            version=f"version_{args_version}",
            name=args_log_dir,
            log_graph=True,
        )
        log_path = f"{args_log_dir}/version_{args_version}"
        return logger, log_path
    elif logger_method == "wandb":
        log_path = f"{args_log_dir}/version_{args_version}"
        wandb_path = f"{log_path}/wandb"
        if not os.path.exists(wandb_path):
            os.makedirs(wandb_path)
        run_uid = args_uid if args_uid else wandb.util.generate_id()
        logger = WandbLogger(
            project="dvil",
            save_dir=log_path,
            version=f"version_{args_version}_{run_uid}",
            name=f"dvil_{args_version}_{run_uid}",
            log_model="all",
        )
        return logger, wandb_path
    else:
        raise NotImplementedError(f"Unknown logger method: {logger_method}")


def train(args: argparse.Namespace) -> None:
    # define how the model is loaded in the prepare_model.py file
    model = prepare_model(args.__dict__)
    train_dl, val_dl = prepare_datasets(args.__dict__)
    logger, log_path = prepare_logger(args.__dict__)

    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_ckpt = ModelCheckpoint(
        dirpath=f"{log_path}/checkpoints",
        monitor="val_f1",
        filename=f"{args.pre + '-' if args.pre != '' else ''}{{epoch:02d}}-{{val_f1:.4f}}-{{val_jaccard:.4f}}",
        verbose=True,
        save_last=True,
        save_top_k=2,
        mode="max",
    )
    callbacks = [ModelSummary(-1), TQDMProgressBar(refresh_rate=1), model_ckpt] + (
        [] if args.fast_dev_run else [lr_monitor]
    )
    if args.fast_dev_run:
        num_gpus = 1
    else:
        num_gpus = "auto" if args.gpus == -1 else args.gpus
    trainer = Trainer(
        accelerator="auto",
        # strategy=DDPStrategy(find_unused_parameters=True),
        devices=num_gpus,
        max_epochs=args.max_epochs,
        logger=logger,
        profiler=None,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run,
    )
    if isinstance(logger, WandbLogger):
        logger.watch(model, log="all", log_freq=100)
    trainer.fit(model, train_dl, val_dl, ckpt_path=args.prev_ckpt if args.resume else None)


def parse_args(args: argparse.Namespace) -> argparse.Namespace:
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"config file does not exist: {args.config}")
    if args.prev_ckpt and not os.path.exists(args.prev_ckpt):
        raise FileNotFoundError(f"previous checkpoint file does not exist: {args.prev_ckpt}")
    if args.log_dir and not os.path.isdir(args.log_dir):
        print(f"Log dir does not exist: {args.log_dir}. Trying to create it..")
        os.makedirs(args.log_dir)
    if args.resume and not args.prev_ckpt:
        raise ValueError("resume is true but there's no checkpoint specified")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
        args.model_args = config["model_args"]
        args.training_args = config["training_args"]
        args.max_epochs = config["training_args"]["max_epochs"]

    rich.print_json(data=args.__dict__)
    return args


def main():
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        help="the path to a config file",
        required=True,
    )
    parser.add_argument(
        "-p",
        "--prev-ckpt",
        type=str,
        help="the path to a previous checkpoint",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--resume",
        action="store_true",
        help="resume the training progress? False will reset the optimizer state (True/False)",
    )
    parser.add_argument(
        "-v",
        "--version",
        type=str,
        help="the version of this model (same as the one saved in log dir)",
        default="0",
    )
    parser.add_argument(
        "-l",
        "--log-dir",
        type=str,
        help="the path to the log directory",
        default="lightning_logs/dvil",
    )
    parser.add_argument(
        "-f",
        "--fast-dev-run",
        action="store_true",
        help="fast dev run? (True/Fase)",
    )
    parser.add_argument(
        "--pre",
        type=str,
        help="checkpoint's prefix",
        default="",
    )
    parser.add_argument(
        "--gpus",
        type=int,
        help="number of gpus to use (-1 for all available gpus)",
        default=-1,
    )
    parser.add_argument(
        "--logger",
        type=str,
        choices=["tensorboard", "wandb"],
        help="logger method (tensorboard/wandb)",
        default="tensorboard",
    )
    parser.add_argument(
        "--uid",
        type=str,
        help="unique id for wandb in case resuming runs with same version",
        default=None,
    )
    args = parser.parse_args()
    args = parse_args(args)

    train(args)


if __name__ == "__main__":
    main()
