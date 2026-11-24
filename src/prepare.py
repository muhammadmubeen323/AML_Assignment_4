"""Create a subject-independent validation split and normalized tensors."""

import os
import numpy as np
import torch

from load_data import load_raw, CHANNELS

VAL_SUBJECTS = [3, 15, 25, 30]


def prepare(data_dir="data", results_dir="results"):
    """Load data, split by volunteer, standardize channels, and save metadata.

    Statistics are fitted only on training volunteers to avoid validation or
    test information leaking into preprocessing.
    """
    raw = load_raw(data_dir)
    train_raw = raw["train"]
    # Reserve complete volunteers rather than random overlapping windows.
    is_val = np.isin(train_raw["subject"], VAL_SUBJECTS)

    X_train_raw = train_raw["X"][~is_val]
    y_train_raw = train_raw["y"][~is_val]
    X_val_raw = train_raw["X"][is_val]
    y_val_raw = train_raw["y"][is_val]
    X_test_raw = raw["test"]["X"]
    y_test_raw = raw["test"]["y"]

    # Pool training samples and time steps while keeping one statistic/channel.
    mean = X_train_raw.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = X_train_raw.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    # Guard against division by zero for any constant sensor channel.
    std = np.where(std > 1e-8, std, 1.0)

    X_train = (X_train_raw - mean) / std
    X_val = (X_val_raw - mean) / std
    X_test = (X_test_raw - mean) / std

    # UCI activity IDs are 1..6; PyTorch cross entropy expects 0..5.
    tensors = {
        "X_train": torch.from_numpy(X_train.astype(np.float32)),
        "y_train": torch.from_numpy((y_train_raw - 1).astype(np.int64)),
        "X_val": torch.from_numpy(X_val.astype(np.float32)),
        "y_val": torch.from_numpy((y_val_raw - 1).astype(np.int64)),
        "X_test": torch.from_numpy(X_test.astype(np.float32)),
        "y_test": torch.from_numpy((y_test_raw - 1).astype(np.int64)),
        "subject_train": train_raw["subject"][~is_val],
        "subject_val": train_raw["subject"][is_val],
        "subject_test": raw["test"]["subject"],
        "activity_names": raw["activity_names"],
        "mean": torch.from_numpy(mean),
        "std": torch.from_numpy(std),
    }
    os.makedirs(results_dir, exist_ok=True)
    # Keep fitted preprocessing values and subject assignments reproducible.
    np.savez(
        os.path.join(results_dir, "prepared.npz"),
        mean=mean,
        std=std,
        train_subjects=np.unique(train_raw["subject"][~is_val]),
        val_subjects=np.unique(train_raw["subject"][is_val]),
        test_subjects=np.unique(raw["test"]["subject"]),
    )
    return tensors


def print_data_report(tensors):
    """Print split sizes, class counts, and fitted channel statistics."""
    names = tensors["activity_names"]
    print(
        "X_train",
        tuple(tensors["X_train"].shape),
        "y_train",
        tuple(tensors["y_train"].shape),
    )
    print(
        "X_val", tuple(tensors["X_val"].shape), "y_val", tuple(tensors["y_val"].shape)
    )
    print(
        "X_test",
        tuple(tensors["X_test"].shape),
        "y_test",
        tuple(tensors["y_test"].shape),
    )
    print("activity counts:")
    for split in ("train", "val", "test"):
        counts = torch.bincount(tensors[f"y_{split}"], minlength=6).tolist()
        print(split, dict(zip(names, counts)))

    mean = tensors["mean"].numpy()
    std = tensors["std"].numpy()
    print("channel mean and std:")
    for i, channel in enumerate(CHANNELS):
        print(f"{channel}: mean={mean[i]:.8f}, std={std[i]:.8f}")


if __name__ == "__main__":
    # Prepare the data and display the key split/preprocessing facts.
    tensors = prepare()
    print_data_report(tensors)
