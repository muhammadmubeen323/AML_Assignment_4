"""Load the UCI HAR raw inertial signals without using engineered features."""

from pathlib import Path
import numpy as np

# Keep a single channel order throughout data loading, training, and the app.
CHANNELS = [
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
]


def _find_root(data_dir):
    """Find the extracted dataset root, ignoring macOS metadata copies."""
    for d in sorted(Path(data_dir).rglob("Inertial Signals")):
        if d.parent.name == "train":
            if "__MACOSX" in d.parts:
                continue
            return d.parent.parent
    raise FileNotFoundError(
        f"No 'train/Inertial Signals' folder under {data_dir}. "
        "Unzip the dataset (and any zip inside it) first."
    )


def load_raw(data_dir="data"):
    """Read each split as (windows, time steps, channels) plus labels/subjects.

    The label file uses one-based activity IDs; conversion to zero-based
    classes is deliberately handled later by the preparation step.
    """
    root = _find_root(data_dir)
    out = {}
    for split in ("train", "test"):
        sig = root / split / "Inertial Signals"
        # Stack channel files last so each window has shape (128, 9).
        X = np.stack(
            [
                np.loadtxt(sig / f"{channel}_{split}.txt", dtype=np.float32)
                for channel in CHANNELS
            ],
            axis=-1,
        )
        y = np.loadtxt(root / split / f"y_{split}.txt", dtype=int)
        subject = np.loadtxt(root / split / f"subject_{split}.txt", dtype=int)
        out[split] = {"X": X, "y": y, "subject": subject}

    names = [
        line.split()[1]
        for line in (root / "activity_labels.txt").read_text().splitlines()
        if line.strip()
    ]
    out["activity_names"] = names
    return out


if __name__ == "__main__":
    # Print a compact sanity check when this file is run directly.
    raw = load_raw()
    for split in ("train", "test"):
        d = raw[split]
        print(
            f"{split}: X {d['X'].shape}, y {d['y'].shape}, subjects "
            f"{sorted(set(d['subject'].tolist()))}"
        )
    print("activities:", raw["activity_names"])
