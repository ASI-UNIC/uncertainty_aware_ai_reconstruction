import os
import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import (
    ModelCheckpoint,
)
import pickle

from model import get_model
from dataset import get_dataloaders


class TimeSeriesLightningModule(pl.LightningModule):
    """
    PyTorch Lightning module for time series models with perturbation conditioning.
    """

    def __init__(
        self,
        model_type="hybrid",
        compressed_dim=512,
        sparsity=2,
        input_dim=2000,
        case_index=0,
        num_layers=2,
        dim_feedforward=1024,
        seq_len=16,
        normalize=True,
        output_dim=2000,
        hidden_dim=256,
        dropout=0.3,
        task="regression",
        lr=0.001,
        weight_decay=1e-5,
        get_strips=False,
        **kwargs,
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Create model
        self.model = get_model(
            model_type=model_type,
            num_layers=num_layers,
            compressed_dim=compressed_dim,
            normalize=normalize,
            seq_len=seq_len,
            dim_feedforward=dim_feedforward,
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            task=task,
            **kwargs,
        )

        # Loss function
        if task == "regression" or task == "interpolation":
            self.criterion = nn.MSELoss()  # nn.MSELoss()
        else:
            self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.task = task

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        if self.task == "regression" or self.task == "interpolation":
            sequences, targets = batch
            sequences = sequences[:, :, : self.model.input_dim]
            targets = targets[:, :, : self.model.input_dim]
            outputs = self(sequences)
            loss = self.criterion(outputs, targets)
            sequence_loss = self.criterion(sequences, targets)

            # Log metrics
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log(
                "train_seq",
                sequence_loss,
                on_step=True,
                on_epoch=True,
                prog_bar=True,
            )

            return loss
        else:
            sequences, labels = batch
            labels = labels.squeeze()
            outputs = self(sequences)
            loss = self.criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float(f).mean()

            # Log metrics
            self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
            self.log("train_acc", accuracy, on_step=True, on_epoch=True, prog_bar=True)

            return loss

    def validation_step(self, batch, batch_idx):
        if self.task == "regression" or self.task == "interpolation":
            sequences, targets = batch
            sequences = sequences[:, :, : self.model.input_dim]
            targets = targets[:, :, : self.model.input_dim]
            outputs = self(sequences)
            loss = self.criterion(outputs, targets)
            sequence_loss = self.criterion(sequences, targets)

            # Log metrics
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log(
                "val_seq",
                sequence_loss,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
            )

            return loss
        else:
            sequences, labels = batch
            labels = labels.squeeze()
            outputs = self(sequences)
            loss = self.criterion(outputs, labels)

            # Calculate accuracy
            _, predicted = outputs.max(1)
            accuracy = (predicted == labels).float().mean()

            # Log metrics
            self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("val_acc", accuracy, on_step=False, on_epoch=True, prog_bar=True)

            return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            # weight_decay=self.hparams.weight_decay,
        )

        return {
            "optimizer": optimizer,
        }


class TimeSeriesDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for time series data.
    """

    def __init__(
        self,
        data_dir,
        batch_size=32,
        seq_len=100,
        step_size=2,
        prediction_horizon=1,
        normalize=True,
        train_ratio=0.8,
        task="regression",
        input_dim=100,
        num_workers=4,
        sparsity=2,
        interpolation_method="cubic",
        start_dim=0,
        case_index=0,
        get_strips=False,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.step_size = step_size
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.train_ratio = train_ratio
        self.task = task
        self.num_workers = num_workers
        self.sparsity = sparsity
        self.interpolation_method = interpolation_method
        self.scaler = None
        self.input_dim = input_dim
        self.start_dim = start_dim
        self.case_index = case_index
        self.get_strips = get_strips

    def setup(self, stage=None):
        # Create dataloaders
        (
            self.train_loader,
            self.global_min,
            self.global_max,
            self.global_mean,
            self.global_std,
        ) = get_dataloaders(
            data_dir=self.data_dir,
            batch_size=self.batch_size,
            input_dim=self.input_dim,
            seq_len=self.seq_len,
            step_size=self.step_size,
            prediction_horizon=self.prediction_horizon,
            normalize=self.normalize,
            train_ratio=self.train_ratio,
            task=self.task,
            num_workers=self.num_workers,
            sparsity=self.sparsity,
            interpolation_method=self.interpolation_method,
            start_dim=self.start_dim,
            case_index=self.case_index,
            get_strips=self.get_strips,
        )

    def train_dataloader(self):
        return self.train_loader

    def val_dataloader(self):
        return self.val_loader

    def get_scaler(self):
        return self.scaler


def train(args, start_dim):
    """Main training function using PyTorch Lightning."""

    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Create data module
    print("\nInitializing data module...")
    data_module = TimeSeriesDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        step_size=args.step_size,
        prediction_horizon=args.prediction_horizon,
        normalize=args.normalize,
        train_ratio=args.train_ratio,
        task=args.task,
        input_dim=args.input_dim,
        num_workers=args.num_workers,
        sparsity=args.sparsity,
        start_dim=start_dim,
        case_index=args.case_index,
        get_strips=args.get_strips,
    )

    # Setup data to get scaler
    data_module.setup()

    # Save scaler
    if data_module.scaler is not None:
        scaler_path = os.path.join(args.output_dir, "scaler.pkl")
        with open(scaler_path, "wb") as f:
            pickle.dump(data_module.scaler, f)
        print(f"Saved scaler to {scaler_path}")

    # Create model
    print("\nInitializing model...")
    if args.task == "regression" or args.task == "interpolation":
        output_dim = args.input_dim if args.input_dim > 0 else 2000
    else:
        output_dim = 4  # Number of classes

    model = TimeSeriesLightningModule(
        model_type=args.model_type,
        input_dim=args.input_dim,
        sparsity=args.sparsity,
        case_index=args.case_index,
        output_dim=output_dim,
        dim_feedforward=args.dim_feedforward,
        num_layers=args.num_layers,
        compressed_dim=args.compressed_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
        seq_len=args.seq_len,
        step_size=args.step_size,
        normalize=args.normalize,
        task=args.task,
        lr=args.lr,
        weight_decay=args.weight_decay,
        get_strips=args.get_strips,
    )

    print(f"Model: {args.model_type}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(
            args.output_dir,
            str(args.get_strips),
            str(args.seed),
            "case " + str(args.case_index),
            str(start_dim),
            "sparsity " + str(args.sparsity),
            "seq_len " + str(args.seq_len),
        ),
        filename="{epoch}-{val_loss:.4f}",
        monitor="train_loss",
        mode="min",
        save_top_k=3,
        save_last=True,
        verbose=True,
    )

    # early_stop_callback = EarlyStopping(
    #     monitor="val_loss", patience=args.patience, mode="min", verbose=True
    # )

    # lr_monitor = LearningRateMonitor(logging_interval="epoch")

    # Trainer
    trainer = pl.Trainer(
        num_sanity_val_steps=0,
        max_epochs=args.epochs,
        accelerator="cpu",  # Automatically use GPU if available
        devices=args.devices,
        callbacks=[checkpoint_callback],
        # gradient_clip_val=args.gradient_clip_val,
        # log_every_n_steps=10,
        deterministic=True,
        # precision=args.precision,
        # accumulate_grad_batches=args.accumulate_grad_batches,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(
        model,
        train_dataloaders=data_module.train_dataloader(),
    )

    print("\nTraining completed!")
    print(f"Best model saved to: {checkpoint_callback.best_model_path}")
    print(f"Best validation loss: {checkpoint_callback.best_model_score:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train perturbation-conditioned time series model with PyTorch Lightning"
    )

    # Data parameters
    parser.add_argument(
        "--data_dir", type=str, default="data/mean_strips", help="Data directory"
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs", help="Output directory"
    )
    parser.add_argument("--seq_len", type=int, default=16, help="Sequence length")
    parser.add_argument("--step_size", type=int, default=2, help="Step size")
    parser.add_argument(
        "--prediction_horizon", type=int, default=0, help="Prediction horizon"
    )
    parser.add_argument(
        "--normalize", action="store_true", default=True, help="Normalize data"
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="Train/val split ratio"
    )
    parser.add_argument(
        "--case_index", type=int, default=0, help="Case index to use for training"
    )
    # Add to train.py argument parser (around line 305):

    # Add these arguments in the "Model parameters" section:
    parser.add_argument(
        "--dim_feedforward",
        type=int,
        default=32,
        help="Feedforward dimension in the model",
    )
    parser.add_argument(
        "--num_layers",
        type=int,
        default=1,
        help="Number of layers in the model",
    )
    parser.add_argument(
        "--compressed_dim",
        type=int,
        default=32,
        help="Compressed dimension in the model",
    )
    parser.add_argument("--hidden_dim", type=int, default=32, help="Hidden dimension")
    parser.add_argument("--start_dim", type=int, default=0)
    parser.add_argument(
        "--input_dim", type=int, default=100, help="Perturbation embedding dimension"
    )
    parser.add_argument(
        "--sparsity",
        type=int,
        default=8,
        help="Sparsity level for interpolation task (e.g., 5 = keep every 5th timestep)",
    )
    parser.add_argument(
        "--interpolation_method",
        type=str,
        default="cubic",
        choices=["cubic", "linear"],
        help="Interpolation method for sparse data",
    )
    # Model parameters
    parser.add_argument(
        "--model_type",
        type=str,
        default="transformer",
        choices=["hybrid", "lstm", "transformer"],
        help="Model type",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="interpolation",
        choices=["regression", "classification", "interpolation"],
        help="Task type",
    )

    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--get_strips", type=bool, default=False, help="Get strips")

    # Training parameters
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=32, help="Number of workers")
    parser.add_argument(
        "--patience", type=int, default=5, help="Early stopping patience"
    )
    parser.add_argument(
        "--gradient_clip_val", type=float, default=1.0, help="Gradient clipping value"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=1,
        help="Gradient accumulation steps",
    )

    # PyTorch Lightning specific
    parser.add_argument(
        "--devices", type=int, default=1, help="Number of devices to use"
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32",
        choices=["16", "32", "bf16"],
        help="Training precision",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--version", type=str, default=None, help="Experiment version")

    args = parser.parse_args()

    if args.task == "interpolation":
        assert args.sparsity is not None

    # for i in range(args.start_dim, 74112, args.input_dim):
    for case_index in [0, 1, 2, 3]:
        args.case_index = case_index
        for sparsity in [2, 4, 8, 16, 64]:
            args.sparsity = sparsity
            for seq_len in [4, 8, 16, 32]:
                args.seq_len = seq_len
                train(args, args.start_dim)
