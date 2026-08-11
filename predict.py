import numpy as np
import pickle
import torch
import argparse
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MultipleLocator
from copy import deepcopy
from sklearn.metrics import (
    r2_score,
    mean_absolute_percentage_error,
    mean_absolute_error,
)

from train import TimeSeriesLightningModule, TimeSeriesDataModule


plt.rcParams.update({"font.size": 22})
plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18
mpl.rcParams["legend.fontsize"] = 18
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False


def load_lightning_model(checkpoint_path, device="cpu"):
    """Load trained PyTorch Lightning model."""
    model = TimeSeriesLightningModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    model.to(device)
    return model


def predict(
    model, sequence, targets, perturbation, global_mean, global_std, device="cpu"
):
    """
    Make prediction for a single sequence.

    Args:
        model: Trained model
        sequence: Input sequence (seq_len, input_dim)
        targets: Target sequence (seq_len, input_dim)
        perturbation: Perturbation value (float)
        global_mean: Global mean of the dataset
        global_std: Global standard deviation of the dataset
        device: Device to use

    Returns:
        prediction: Model output
    """
    case_shapes = [len(case) for case in sequence]

    # Convert to tensors
    sequence = torch.FloatTensor(sequence)
    targets = torch.FloatTensor(targets)
    sequence = sequence.reshape(
        sequence.shape[0] * sequence.shape[1], *sequence.shape[2:]
    ).to(device)
    targets = targets.reshape(
        targets.shape[0] * targets.shape[1], *targets.shape[2:]
    ).to(device)
    perturbation = torch.FloatTensor([[[perturbation]]]).to(device)  # (1, 1)
    perturbation = perturbation.repeat(sequence.shape[0], 1, 1)

    # Predict
    with torch.no_grad():
        output = model(sequence)

    # Convert to numpy
    output = output.cpu().numpy()
    numpy_targets = targets.cpu().numpy()

    all_scaled_outputs = []
    all_scaled_sequences = []
    all_scaled_targets = []

    for i, case_shape in enumerate(case_shapes):
        current_output = output[:case_shape]
        current_targets = numpy_targets[:case_shape]
        current_sequences = sequence[:case_shape]

        original_shape = current_output.shape
        current_output = current_output.reshape(-1, original_shape[-1])
        current_targets = current_targets.reshape(-1, original_shape[-1])
        current_sequences = current_sequences.reshape(-1, original_shape[-1])

        scaled_outputs = (
            current_output * global_std[: current_output.shape[1]]
            + global_mean[: current_output.shape[1]]
        )
        scaled_targets = (
            current_targets * global_std[: current_targets.shape[1]]
            + global_mean[: current_targets.shape[1]]
        )
        scaled_sequences = (
            current_sequences * global_std[: current_sequences.shape[1]]
            + global_mean[: current_sequences.shape[1]]
        )

        all_scaled_outputs.append(scaled_outputs)
        all_scaled_targets.append(scaled_targets)
        all_scaled_sequences.append(scaled_sequences)

    return all_scaled_outputs, all_scaled_sequences, all_scaled_targets


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make predictions with trained Lightning model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to checkpoint",
        default="/Users/darylfung/Documents/Work/Nicosia/hypersonic/outputs/False/case 0/0/sparsity 2/seq_len 4/last.ckpt",
    )
    parser.add_argument("--sparsity", type=int, default=2)
    parser.add_argument(
        "--perturbation",
        type=float,
        help="Perturbation value",
        default=0.13,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    sparsity_metric = {
        "sparsity 2": {},
        "sparsity 4": {},
        "sparsity 8": {},
        "sparsity 16": {},
        "sparsity 32": {},
        "sparsity 64": {},
    }
    case_metric = {
        "case 0": deepcopy(sparsity_metric),
        "case 1": deepcopy(sparsity_metric),
        "case 2": deepcopy(sparsity_metric),
        "case 3": deepcopy(sparsity_metric),
    }

    metrics = {
        "error": deepcopy(case_metric),
        "r2": deepcopy(case_metric),
        "mape": deepcopy(case_metric),
        "mae": deepcopy(case_metric),
        "max_mae": deepcopy(case_metric),
    }

    for case in ["case 0", "case 1", "case 2", "case 3"]:
        for sparsity in [2, 4, 8, 16, 32, 64]:
            for seq_len in [4, 8, 16, 32]:
                current_dir = f"visual/False/{sparsity}/{seq_len}"
                os.makedirs(current_dir, exist_ok=True)
                checkpoint = (
                    args.checkpoint.replace("sparsity 2", f"sparsity {sparsity}")
                    .replace("seq_len 4", f"seq_len {seq_len}")
                    .replace("case 0", case)
                )
                # Load model
                print("Loading model...")
                model = load_lightning_model(checkpoint, device)

                print(f"Model type: {model.hparams.model_type}")
                print(f"Task: {model.hparams.task}")
                print(f"Input dim: {model.hparams.input_dim}")
                print(f"Output dim: {model.hparams.output_dim}")

                # Load the dataset
                print("\nLoading dataset...")
                data_dir = "data/mean_strips"  # Update this with your data directory if different

                # Get the data module with the same parameters as training
                data_module = TimeSeriesDataModule(
                    data_dir=data_dir,
                    batch_size=1,  # Predict one sequence at a time
                    seq_len=model.hparams.seq_len
                    if hasattr(model.hparams, "seq_len")
                    else 16,
                    step_size=model.hparams.step_size
                    if hasattr(model.hparams, "step_size")
                    else 2,
                    normalize=model.hparams.normalize
                    if hasattr(model.hparams, "normalize")
                    else True,
                    task=model.hparams.task,
                    sparsity=model.hparams.sparsity,
                    interpolation_method=model.hparams.interpolation_method
                    if hasattr(model.hparams, "interpolation_method")
                    else "cubic",
                    input_dim=model.hparams.input_dim,
                    start_dim=0,
                    case_index=model.hparams.case_index,
                    get_strips=False,
                )

                # Setup the data module (this will load and process the data)
                data_module.setup(stage="predict")

                # Get the training dataloader
                sequences = data_module.train_dataloader().dataset.sequences
                targets = data_module.train_dataloader().dataset.targets
                perturbations = data_module.train_dataloader().dataset.pert_values

                # Use the first sequence from the batch
                # sequences = np.array(sequences)

                # Make prediction
                print(f"\nMaking prediction with perturbation={args.perturbation}...")
                predictions, sequences, targets = predict(
                    model,
                    sequences,
                    targets,
                    args.perturbation,
                    data_module.global_mean,
                    data_module.global_std,
                    device,
                )

                for i in range(len(predictions)):
                    error = (predictions[i] - targets[i]) ** 2
                    error = error.mean()
                    sequences_error = (sequences[i] - targets[i]) ** 2
                    sequences_error = sequences_error.mean()
                    print(f"Error: {error}")
                    print(f"Sequences error: {sequences_error}")
                    r2 = r2_score(targets[i], predictions[i])
                    sequences_r2 = r2_score(targets[i], sequences[i])
                    print(f"R2: {r2}")
                    print(f"Sequences R2: {sequences_r2}")
                    mape = (
                        mean_absolute_percentage_error(targets[i], predictions[i]) * 100
                    )
                    sequences_mape = (
                        mean_absolute_percentage_error(targets[i], sequences[i]) * 100
                    )
                    print(f"MAPE: {mape}")
                    print(f"Sequences MAPE: {sequences_mape}")
                    mae = mean_absolute_error(targets[i], predictions[i])
                    sequences_mae = mean_absolute_error(targets[i], sequences[i])
                    max_mae = np.max(np.abs(targets[i] - predictions[i]))
                    print(f"MAE: {mae}")
                    print(f"Sequences MAE: {sequences_mae}")

                    plt.plot(predictions[i][:, 0], label="Prediction")
                    plt.plot(targets[i][:, 0], label="Target")
                    plt.legend(frameon=False)
                    plt.xlabel("Features")
                    plt.ylabel("Pressure")
                    plt.savefig(
                        f"{current_dir}/predictions_{case}.png",
                        bbox_inches="tight",
                    )
                    plt.clf()
                    plt.close()

                    plt.plot(sequences[i][:, 0], label="Sequence")
                    plt.plot(predictions[i][:, 0], label="Prediction")
                    plt.plot(targets[i][:, 0], label="Target")
                    plt.xlabel("Features")
                    plt.ylabel("Pressure")
                    plt.legend(frameon=False)
                    plt.savefig(
                        f"{current_dir}/predictions_{case}.with_sequences.png",
                        bbox_inches="tight",
                    )
                    plt.clf()
                    plt.close()

                    metrics["error"][case][f"sparsity {sparsity}"][
                        f"seq_len {seq_len}"
                    ] = error
                    metrics["r2"][case][f"sparsity {sparsity}"][
                        f"seq_len {seq_len}"
                    ] = r2
                    metrics["mape"][case][f"sparsity {sparsity}"][
                        f"seq_len {seq_len}"
                    ] = mape
                    metrics["mae"][case][f"sparsity {sparsity}"][
                        f"seq_len {seq_len}"
                    ] = mae
                    metrics["max_mae"][case][f"sparsity {sparsity}"][
                        f"seq_len {seq_len}"
                    ] = max_mae

    pickle.dump(metrics, open("metrics.pkl", "wb"))

    metrics = pickle.load(open("metrics.pkl", "rb"))

    for case in ["case 0", "case 1", "case 2", "case 3"]:
        for error in ["error", "r2", "mape", "mae"]:
            if error == "error":
                error_text = "MSE"
            else:
                error_text = error

            for sparsity in [2, 4, 8, 16, 32, 64]:
                current_dir = f"visual/{case}/{sparsity}"
                os.makedirs(current_dir, exist_ok=True)
                for seq_len in [4, 8, 16, 32]:
                    plt.bar(
                        f"{seq_len}",
                        metrics[error][case][f"sparsity {sparsity}"][
                            f"seq_len {seq_len}"
                        ],
                    )
                plt.title(f"{error} for {case} with sparsity {sparsity}")
                plt.legend(frameon=False)
                plt.xlabel("Sequence Length")
                plt.ylabel(f"{error_text.upper()}")
                plt.gca().xaxis.set_minor_locator(MultipleLocator(5))
                plt.savefig(
                    f"{current_dir}/{error}_{case}.png",
                    bbox_inches="tight",
                )
                plt.clf()
                plt.close()

    # print in latex for the errors to create a table
    for case in ["case 0", "case 1", "case 2", "case 3"]:
        print(case)
        print("==============================")
        text = f"""\\begin{{table}}[h]
\\centering
\\caption{{ {case}: Error metrics for different sparsity levels and sequence lengths}}
\\label{{tab:{case}_all_metrics}}
\\begin{{tabular}}{{cc|cccc}}
\\hline
\\textbf{{Sparsity}} & \\textbf{{Seq}} & \\textbf{{MAE}} & \\textbf{{MAPE}} & \\textbf{{Max MAE}} & \\textbf{{Average MAPE}}  \\\\
\\hline
"""
        for sparsity in [2, 4, 8, 16, 32, 64]:
            average_mape = []
            for seq_len in [4, 8, 16, 32]:
                text += f"{sparsity} & {seq_len} "
                for error in ["mae", "mape", "max_mae", "average_mape"]:
                    if error == "average_mape":
                        text += "& "
                        average_mape.append(
                            metrics["mape"][case][f"sparsity {sparsity}"][
                                f"seq_len {seq_len}"
                            ]
                        )
                        continue
                    error_text = error
                    text += f"& {round(metrics[error][case][f'sparsity {sparsity}'][f'seq_len {seq_len}'], 2)} "

                if seq_len == 32:
                    text += f" {np.mean(average_mape):.2f}\\\\ \n"
                else:
                    text += "\\\\ \n"
            text += "\\hline \n"
        text += "\\end{tabular} \n"
        text += "\\end{table} \n"
        print(text)
