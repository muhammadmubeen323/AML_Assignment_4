"""Plot example raw inertial windows, one example per activity class."""

import os
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from load_data import load_raw, CHANNELS

FS = 50


def plot_one_per_activity(split, names, out_path="results/example_windows.png"):
    """Plot body acceleration and gyroscope channels for each activity."""
    t = np.arange(128) / FS
    fig, axes = plt.subplots(2, 6, figsize=(20, 5), sharex=True)
    for k, name in enumerate(names):
        i = int(np.flatnonzero(split["y"] == k + 1)[0])
        for c in range(3):
            axes[0, k].plot(t, split["X"][i, :, c], label=CHANNELS[c])
            axes[1, k].plot(t, split["X"][i, :, 3 + c], label=CHANNELS[3 + c])
        axes[0, k].set_title(name, fontsize=9)
        axes[1, k].set_xlabel("time (s)")
    axes[0, 0].set_ylabel("body acc (g)")
    axes[1, 0].set_ylabel("gyro (rad/s)")
    axes[0, 0].legend(fontsize=7)
    axes[1, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    print(f"saved {out_path}")


if __name__ == "__main__":
    # Create the output folder before rendering the training examples.
    os.makedirs("results", exist_ok=True)
    raw = load_raw()
    plot_one_per_activity(raw["train"], raw["activity_names"])
