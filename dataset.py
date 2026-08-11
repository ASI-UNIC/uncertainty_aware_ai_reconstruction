import pickle
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.interpolate import CubicSpline


class PerturbationTimeSeriesDataset(Dataset):
    """
    Dataset for loading time series data with perturbation values.
    Supports sparse sampling with spline interpolation for interpolation refinement tasks.

    Args:
        data_dir: Directory containing CSV files
        seq_len: Length of input sequences (default: 100)
        prediction_horizon: Number of timesteps to predict (default: 1)
        normalize: Whether to normalize the data (default: True)
        train: Whether this is training data (default: True)
        train_ratio: Ratio of training data (default: 0.8)
        task: 'regression' or 'classification' or 'interpolation'
        sparsity: Sparsity level for interpolation task (e.g., 5 means keep every 5th timestep)
        interpolation_method: 'cubic' or 'linear' for spline interpolation
    """

    def __init__(
        self,
        data_list: list,
        perturbation_list: list,
        global_min: float,
        global_max: float,
        global_mean: float,
        global_std: float,
        seq_len=100,
        step_size=2,
        input_dim=100,
        prediction_horizon=0,
        normalize=True,
        train=True,
        train_ratio=0.8,
        task="regression",
        sparsity=None,
        interpolation_method="cubic",
        get_strips=False,
    ):
        self.data_list = data_list
        self.perturbation_list = perturbation_list
        self.global_min = global_min
        self.global_max = global_max
        self.global_mean = global_mean
        self.global_std = global_std
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.step_size = step_size
        self.prediction_horizon = prediction_horizon
        self.normalize = normalize
        self.train = train
        self.task = task
        self.sparsity = sparsity
        self.interpolation_method = interpolation_method
        self.get_strips = get_strips

        os.makedirs(f"intermediate_files/{self.sparsity}", exist_ok=True)

        if os.path.exists(
            f"intermediate_files/{self.sparsity}/{self.train}_scalers.npy"
        ):
            self._load_existing_files()
            return

        # Process data
        self._process_data()

        if self.normalize:
            self.data = [
                self._normalize_array(
                    d, global_min, global_max, global_mean, global_std
                )
                for d in self.data_list
            ]
        else:
            self.data = [d for d in self.data_list]

        # Create sequences
        self.sequences = []
        self.case_sequences = []
        self.targets = []
        self.case_targets = []
        self.pert_values = []

        for i, (data, pert) in enumerate(
            zip(
                self.data,
                self.pert,
            )
        ):
            seqs, tgts, case_seqs, case_tgts = self._create_sequences(data)
            self.sequences.append(seqs)
            self.case_sequences.extend(case_seqs)
            self.targets.append(tgts)
            self.case_targets.extend(case_tgts)
            self.pert_values.append([pert] * len(seqs))

        self.case_sequences = torch.FloatTensor(self.case_sequences)
        self.case_targets = torch.FloatTensor(self.case_targets)

        np.save(
            f"intermediate_files/{self.sparsity}/{self.train}_sequences.npy",
            self.sequences,
        )
        np.save(
            f"intermediate_files/{self.sparsity}/{self.train}_case_sequences.npy",
            self.case_sequences,
        )
        np.save(
            f"intermediate_files/{self.sparsity}/{self.train}_targets.npy", self.targets
        )
        np.save(
            f"intermediate_files/{self.sparsity}/{self.train}_case_targets.npy",
            self.case_targets,
        )
        np.save(
            f"intermediate_files/{self.sparsity}/{self.train}_pert_values.npy",
            self.pert_values,
        )

    def _load_existing_files(self):
        """Load existing files."""
        self.sequences = np.load(
            f"intermediate_files/{self.sparsity}/{self.train}_sequences.npy",
            allow_pickle=True,
        )
        self.case_sequences = np.load(
            f"intermediate_files/{self.sparsity}/{self.train}_case_sequences.npy",
            allow_pickle=True,
        )
        self.targets = np.load(
            f"intermediate_files/{self.sparsity}/{self.train}_targets.npy",
            allow_pickle=True,
        )
        self.case_targets = np.load(
            f"intermediate_files/{self.sparsity}/{self.train}_case_targets.npy",
            allow_pickle=True,
        )
        self.case_pert_values = np.load(
            f"intermediate_files/{self.sparsity}/{self.train}_case_pert_values.npy",
            allow_pickle=True,
        )
        self.pert_values = np.load(
            f"intermediate_files/{self.sparsity}/{self.train}_pert_values.npy",
            allow_pickle=True,
        )
        self.scalers = np.load(
            f"intermediate_files/{self.sparsity}/{self.train}_scalers.npy",
            allow_pickle=True,
        )

    def _process_data(self):
        """Split each case into train/val."""
        self.data = []
        self.pert = []

        for i, (data, pert) in enumerate(zip(self.data_list, self.perturbation_list)):
            self.data.append(data)
            self.pert.append(pert)

    def _normalize_array(self, data, global_min, global_max, global_mean, global_std):
        """Normalize a single array."""
        # data_normalized = (data - global_min) / (global_max - global_min)
        data_normalized = (data - global_mean[: data.shape[1]]) / global_std[
            : data.shape[1]
        ]
        return data_normalized

    def _apply_spline_interpolation(self, data):
        """
        Apply cubic spline interpolation to create sparse-to-dense data.

        Args:
            data: Full dense data array (timesteps, features)

        Returns:
            interpolated_data: Data with cubic spline interpolation applied
            sparse_mask: Boolean mask indicating which timesteps are original (True) vs interpolated (False)
        """
        if self.sparsity is None or self.sparsity <= 1:
            # No sparsity, return original data
            return data, data

        num_timesteps, _ = data.shape

        # Create sparse indices (keep every sparsity-th timestep)
        # e.g., sparsity=5 -> keep indices 0, 5, 10, 15, ...
        sparse_indices = np.arange(0, num_timesteps, self.sparsity)

        # Create mask for sparse points
        sparse_mask = np.zeros(num_timesteps, dtype=bool)
        sparse_mask[sparse_indices] = True

        # Interpolate all timesteps using cubic spline
        x = np.arange(num_timesteps)
        y = data[sparse_mask]
        cs = CubicSpline(x[sparse_mask], y)
        interpolated_data = cs(x)

        return data, interpolated_data

    def _create_sequences(self, data):
        """Create input sequences and targets."""
        sequences = []
        targets = []
        case_sequences = []
        case_targets = []

        if self.task == "interpolation" and self.sparsity is not None:
            # For interpolation task
            # Extract sequence
            seq_original = data

            # Apply spline interpolation
            seq_original, seq_interpolated = self._apply_spline_interpolation(
                seq_original
            )
        else:
            raise ValueError(
                "Invalid task or sparsity, only interpolation task is supported for now"
            )

        for i in range(0, len(data) - self.seq_len, self.seq_len):
            current_seq_segment = seq_interpolated[i : i + self.seq_len]
            current_target_segment = seq_original[i : i + self.seq_len]

            sequences.append(current_seq_segment)
            targets.append(current_target_segment)

        for i in range(0, len(data) - self.seq_len, self.step_size):
            current_seq_segment = seq_interpolated[i : i + self.seq_len]
            current_target_segment = seq_original[i : i + self.seq_len]

            case_sequences.append(current_seq_segment)
            case_targets.append(current_target_segment)
            # Input: interpolated sequence
            # Target: original sequence (ground truth)

        # elif self.task == "regression":
        #     # Standard regression: predict next timestep(s)
        #     seq = data[i : i + self.seq_len]
        #     target = data[
        #         i + self.seq_len : i + self.seq_len + self.prediction_horizon
        #     ]
        #     if self.prediction_horizon == 1:
        #         target = target.squeeze(0)

        #     sequences.append(seq)
        #     targets.append(target)

        #     else:
        #         # Classification task - target is the perturbation label
        #         seq = data[i : i + self.seq_len]
        #         sequences.append(seq)
        #         # targets will be None, use case_labels instead

        return sequences, targets, case_sequences, case_targets

    def __len__(self):
        return len(self.case_sequences)

    def __getitem__(self, idx):
        sequence = self.case_sequences[idx]

        if self.task == "regression" or self.task == "interpolation":
            target = self.case_targets[idx]
            return sequence, target
        else:
            label = self.case_labels[idx]
            return sequence, label

    def get_data_for_prediction(self):
        return self.data


def get_global_min_max(data_dir, input_dim, get_strips=False):
    # if os.path.exists("global_min_max.pkl"):
    #     with open("global_min_max.pkl", "rb") as f:
    #         return pickle.load(f)

    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])
    all_data = []
    for csv_file in csv_files:
        file_path = os.path.join(data_dir, csv_file)
        df = pd.read_csv(file_path, header=None)
        if get_strips:
            df = get_data_strips(df.values, input_dim)
        all_data.extend(df.values)

    min = np.min(all_data)
    max = np.max(all_data)
    mean = np.mean(all_data, 0)
    std = np.std(all_data, 0)

    with open("global_min_max.pkl", "wb") as f:
        pickle.dump((min, max, mean, std), f)

    return min, max, mean, std


def get_data_strips(array: np.ndarray, input_dim: int):
    total_z = int(array.shape[-1] / input_dim)
    return array.reshape(array.shape[0], total_z, input_dim).mean(1)


def load_csv_data(data_dir, start_dim, input_dim, file_index=0, get_strips=False):
    """Load all CSV files and extract perturbation values from filenames."""
    data_list = []
    perturbation_list = []

    # Get all CSV files
    csv_files = sorted([f for f in os.listdir(data_dir) if f.endswith(".csv")])

    # pert_value = float(csv_files[file_index].split("_")[0])

    # Load CSV
    file_path = os.path.join(data_dir, csv_files[file_index])
    df = pd.read_csv(file_path, header=None)

    # Convert to numpy array (timesteps x features)
    if get_strips:
        data = get_data_strips(df.values, input_dim)
    else:
        data = df.values[:, start_dim : start_dim + input_dim]

    data_list.append(data)
    perturbation_list.append(0)  # TODO fix this

    print(f"Loaded {csv_files[file_index]}: shape={data.shape}, perturbation={0}")

    return data_list, perturbation_list


def normalize(data, global_min, global_max):
    return (data - global_min) / (global_max - global_min)


def get_dataloaders(
    data_dir,
    scaler=None,
    batch_size=32,
    input_dim=100,
    seq_len=100,
    step_size=2,
    prediction_horizon=1,
    normalize=True,
    train_ratio=0.8,
    task="regression",
    num_workers=4,
    sparsity=None,
    interpolation_method="cubic",
    start_dim=0,
    case_index=0,
    get_strips=False,
):
    """
    Create train and validation dataloaders.

    Args:
        data_dir: Directory containing CSV files
        batch_size: Batch size
        seq_len: Sequence length
        prediction_horizon: Number of timesteps to predict
        normalize: Whether to normalize data
        train_ratio: Train/val split ratio
        task: 'regression', 'classification', or 'interpolation'
        num_workers: Number of workers for data loading
        sparsity: Sparsity level for interpolation (e.g., 5 = keep every 5th timestep)
        interpolation_method: 'cubic' or 'linear'
        case_index: Index of case to use for training

    Returns:
        train_loader, val_loader, scaler
    """
    global_min, global_max, global_mean, global_std = get_global_min_max(
        data_dir, input_dim, get_strips=get_strips
    )

    data_list, perturbation_list = load_csv_data(
        data_dir, start_dim, input_dim, case_index, get_strips=get_strips
    )

    # Create training dataset
    train_dataset = PerturbationTimeSeriesDataset(
        data_list=data_list[:],
        perturbation_list=perturbation_list[:],
        seq_len=seq_len,
        step_size=step_size,
        input_dim=input_dim,
        prediction_horizon=prediction_horizon,
        normalize=normalize,
        train=True,
        train_ratio=train_ratio,
        task=task,
        global_min=global_min,
        global_max=global_max,
        global_mean=global_mean,
        global_std=global_std,
        sparsity=sparsity,
        interpolation_method=interpolation_method,
    )

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
    )

    print("\nDataset Statistics:")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Sequence length: {seq_len}")
    print(f"Task: {task}")
    if sparsity:
        print(f"Sparsity: {sparsity} (keep every {sparsity}th timestep)")
        print(f"Interpolation method: {interpolation_method}")

    return train_loader, global_min, global_max, global_mean, global_std


if __name__ == "__main__":
    get_global_min_max("data")
    # Test the dataset
    data_dir = "data"

    print("=" * 60)
    print("Testing Interpolation Task with Sparsity=5:")
    print("=" * 60)
    train_loader, global_min, global_max = get_dataloaders(
        data_dir=data_dir,
        batch_size=8,
        seq_len=100,
        task="interpolation",
        sparsity=5,
        interpolation_method="cubic",
        num_workers=8,
    )

    # Get a batch
    for batch in train_loader:
        seq, pert, target = batch
        print("\nBatch shapes:")
        print(f"  Input (interpolated): {seq.shape}")
        print(f"  Perturbation: {pert.shape}")
        print(f"  Target (ground truth): {target.shape}")
        print(f"  Perturbation values: {pert.squeeze().tolist()}")

        # Show difference between interpolated and ground truth
        diff = torch.abs(seq - target).mean()
        print(f"  Mean absolute difference (interpolation error): {diff.item():.6f}")
        break

    print("\n" + "=" * 60)
    print("Testing Standard Regression Task:")
    print("=" * 60)
    train_loader_reg, val_loader_reg, train_scalers_reg, val_scalers_reg = (
        get_dataloaders(
            data_dir=data_dir,
            batch_size=8,
            seq_len=100,
            prediction_horizon=1,
            task="regression",
            num_workers=0,
        )
    )

    for batch in train_loader_reg:
        seq, pert, target = batch
        print("\nBatch shapes:")
        print(f"  Sequence: {seq.shape}")
        print(f"  Perturbation: {pert.shape}")
        print(f"  Target: {target.shape}")
        break
