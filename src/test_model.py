"""Evaluate the saved checkpoint once and generate assignment result files."""

import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from prepare import prepare
from model import HARNet
from metrics import accuracy, confusion_matrix, per_class_accuracy

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING UPSTAIRS",
    "WALKING DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING",
]


def save_training_plots():
    """Render the saved training/validation loss and validation accuracy."""
    history = torch.load("results/training_history.pt", weights_only=False)["history"]
    epochs = [row[0] for row in history]
    train_loss = [row[1] for row in history]
    val_loss = [row[2] for row in history]
    val_acc = [row[3] for row in history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, label="Training")
    axes[0].plot(epochs, val_loss, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy loss")
    axes[0].legend()
    axes[1].plot(epochs, val_acc, label="Validation")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig("results/training_curves.png", dpi=150)
    plt.close(fig)


def test_once():
    """Run test inference, report metrics, and save plots/tables/analysis."""
    os.makedirs("results", exist_ok=True)
    tensors = prepare()
    model = HARNet()
    model.load_state_dict(
        torch.load("results/best_har_model.pt", map_location="cpu", weights_only=True)
    )
    model.eval()

    X_test = tensors["X_test"]
    y_test = tensors["y_test"]
    # Batch inference avoids storing large intermediate activations.
    all_logits = []
    all_attention = []
    with torch.no_grad():
        for start in range(0, len(X_test), 64):
            logits, alpha = model(X_test[start : start + 64])
            all_logits.append(logits)
            all_attention.append(alpha)
    logits = torch.cat(all_logits)
    attention = torch.cat(all_attention)
    y_pred = logits.argmax(dim=1)

    # Compute final metrics only after all test predictions are collected.
    test_accuracy = accuracy(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred, 6)
    per_class = per_class_accuracy(confusion)

    print(f"Test accuracy: {test_accuracy:.4f}")
    print("Confusion matrix:")
    print(confusion)
    print("Per-class accuracy:")
    for name, value in zip(ACTIVITY_NAMES, per_class.tolist()):
        print(f"{name}: {value:.4f}")

    per_activity_peak = []
    for activity in range(6):
        activity_attention = attention[y_test == activity]
        per_activity_peak.append(activity_attention.max(dim=1).values.mean().item())

    # Persist predictions and attention for later inspection without retraining.
    np.savez(
        "results/test_predictions.npz",
        y_true=y_test.numpy(),
        y_pred=y_pred.numpy(),
        probabilities=torch.softmax(logits, dim=1).numpy(),
        attention=attention.numpy(),
    )
    save_training_plots()
    plot_confusion(confusion)
    plot_correct_examples(X_test, y_test, y_pred, attention)
    plot_average_attention(y_test, attention)
    write_tables(tensors, confusion, per_class, per_activity_peak)
    write_analysis(confusion, per_class, per_activity_peak)
    return test_accuracy, confusion, per_class, per_activity_peak


def plot_confusion(confusion):
    """Draw and annotate the six-class confusion matrix."""
    matrix = confusion.numpy()
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(6), ACTIVITY_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(6), ACTIVITY_NAMES)
    ax.set_xlabel("Predicted activity")
    ax.set_ylabel("True activity")
    for i in range(6):
        for j in range(6):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > matrix.max() / 2 else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig("results/confusion_matrix.png", dpi=150)
    plt.close(fig)


def plot_correct_examples(X_test, y_test, y_pred, attention):
    """Show one test waveform and attention curve for every true class."""
    t = np.arange(128) / 50.0
    fig, axes = plt.subplots(6, 2, figsize=(14, 16), sharex="col")
    for activity, name in enumerate(ACTIVITY_NAMES):
        candidates = torch.where((y_test == activity) & (y_pred == activity))[0]
        index = (
            int(candidates[0])
            if len(candidates)
            else int(torch.where(y_test == activity)[0][0])
        )
        signal = X_test[index].numpy()
        for channel in range(3):
            axes[activity, 0].plot(
                t, signal[:, channel], label=["X", "Y", "Z"][channel]
            )
        axes[activity, 0].set_ylabel(name)
        axes[activity, 0].grid(alpha=0.2)
        if activity == 0:
            axes[activity, 0].legend(fontsize=7)
        axes[activity, 1].plot(t, attention[index].numpy(), color="tab:orange")
        axes[activity, 1].fill_between(
            t, attention[index].numpy(), alpha=0.25, color="tab:orange"
        )
        axes[activity, 1].set_ylim(bottom=0)
        axes[activity, 1].grid(alpha=0.2)
        if activity == 5:
            axes[activity, 0].set_xlabel("Time (s)")
            axes[activity, 1].set_xlabel("Time (s)")
    axes[0, 0].set_title("Body acceleration (standardized)")
    axes[0, 1].set_title("Attention")
    fig.tight_layout()
    fig.savefig("results/attention_examples.png", dpi=150)
    plt.close(fig)


def plot_average_attention(y_test, attention):
    """Compare mean temporal attention distributions by true activity."""
    t = np.arange(128) / 50.0
    fig, ax = plt.subplots(figsize=(12, 5))
    for activity, name in enumerate(ACTIVITY_NAMES):
        average = attention[y_test == activity].mean(dim=0).numpy()
        ax.plot(t, average, label=name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Mean attention weight")
    ax.legend(ncol=3)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig("results/average_attention.png", dpi=150)
    plt.close(fig)


def write_tables(tensors, confusion, per_class, per_activity_peak):
    """Write dataset, training, per-class, and confusion results as Markdown."""
    training = torch.load("results/training_history.pt", weights_only=False)
    train_windows = tensors["X_train"].shape[0]
    val_windows = tensors["X_val"].shape[0]
    test_windows = tensors["X_test"].shape[0]
    train_subjects = len(set(tensors["subject_train"].tolist()))
    val_subjects = len(set(tensors["subject_val"].tolist()))
    test_subjects = len(set(tensors["subject_test"].tolist()))

    lines = [
        "# Results Tables",
        "",
        "## Table 1 - Data",
        "",
        "| Item | Train | Validation | Test |",
        "|---|---:|---:|---:|",
        f"| Windows | {train_windows} | {val_windows} | {test_windows} |",
        f"| Volunteers | {train_subjects} | {val_subjects} | {test_subjects} |",
        "",
        "| Channel | Mean (training) | Standard deviation (training) |",
        "|---|---:|---:|",
    ]
    for i, channel in enumerate(
        tensors["activity_names"]
        if False
        else [
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
    ):
        lines.append(
            f"| {channel} | {tensors['mean'][i].item():.8f} | {tensors['std'][i].item():.8f} |"
        )

    lines += [
        "",
        "## Table 2 - Network and Training",
        "",
        "| Item | Value |",
        "|---|---:|",
        "| Trainable parameters | 46,982 |",
        f"| Initial loss | {training['initial_loss']:.6f} |",
        f"| Best epoch | {training['best_epoch']} |",
        f"| Training loss at best epoch | {training['best_train_loss']:.6f} |",
        f"| Validation loss at best epoch | {training['best_val_loss']:.6f} |",
        f"| Validation accuracy at best epoch | {training['best_val_accuracy']:.4f} |",
        f"| Training time | {training['elapsed_seconds']:.2f} seconds ({training['hardware']}) |",
        "",
        "Parameter calculation:",
        "`2 × (4×64×9 + 4×64×64 + 4×64) + (64×128 + 64 + 64) + (128×6 + 6) = 46,982`",
        "",
        "## Table 3 - Test Results",
        "",
        "| Activity | Per-class accuracy | Mean max alpha |",
        "|---|---:|---:|",
    ]
    for i, name in enumerate(ACTIVITY_NAMES):
        lines.append(
            f"| {name} | {per_class[i].item():.4f} | {per_activity_peak[i]:.6f} |"
        )

    test_accuracy = float(
        (confusion.diag().sum().float() / confusion.sum().float()).item()
    )
    lines += [
        "",
        f"Test accuracy: **{test_accuracy:.4f}**",
        "",
        "| True \\ Predicted | WALKING | UPSTAIRS | DOWNSTAIRS | SITTING | STANDING | LAYING |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    short_names = ["WALKING", "UPSTAIRS", "DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
    matrix = confusion.tolist()
    for i, name in enumerate(short_names):
        lines.append("| " + name + " | " + " | ".join(str(v) for v in matrix[i]) + " |")

    with open("results/tables.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def write_analysis(confusion, per_class, per_activity_peak):
    """Create the assignment discussion using values from this test run."""
    matrix = confusion.tolist()
    off_diagonal = []
    for i in range(6):
        for j in range(6):
            if i != j:
                off_diagonal.append((matrix[i][j], i, j))
    off_diagonal.sort(reverse=True)
    pair = [
        (ACTIVITY_NAMES[i], ACTIVITY_NAMES[j], count)
        for count, i, j in off_diagonal[:2]
    ]
    moving_peak = np.mean(per_activity_peak[:3])
    still_peak = np.mean(per_activity_peak[3:])

    text = f"""# Analysis

## 6.1 Person-independent validation

Validation subjects 3, 15, 25, and 30 are held out before training. Random window-level splitting would leak highly similar, 50%-overlapping windows across adjacent segments and could let the network recognize a person rather than general activity. Subject-level separation gives an honest estimate of generalization.

## 6.2 Most confused activities

The largest off-diagonal confusion counts are {pair[0][0]} -> {pair[0][1]} ({pair[0][2]} windows) and {pair[1][0]} -> {pair[1][1]} ({pair[1][2]} windows). In the waist-mounted recordings, SITTING and STANDING have weak body motion and similar posture-related acceleration/gyroscope patterns, so they are difficult to separate. Their example-waveforms also show much less dynamic change than walking activities.

## 6.3 Why LAYING is easy

LAYING is strongly separated by the gravity component, especially the total-acceleration vertical channel and the body-acceleration response. The phone orientation is nearly stable in lying posture, producing a consistent acceleration direction that differs from upright and moving postures. In this run, LAYING per-class accuracy is {per_class[5].item():.4f}.

## 6.4 Attention peakiness

The mean maximum attention weight is {moving_peak:.6f} for moving activities and {still_peak:.6f} for still activities. These values are estimates of how concentrated each prediction is. A larger value means the network concentrates evidence in fewer moments; a smaller value means it uses more of the 128-step window. Useful temporal information can occur over several cycles for movement, while stationary activities may depend on persistent orientation information.

## 6.5 Why bidirectional processing is allowed

The test window is a complete offline 2.56-second segment, so every point is available when the network makes a decision. Bidirectional processing is therefore appropriate for this activity-recognition task. It would not be appropriate for a strictly real-time or causal stream (for example, live fall detection), where future sensor samples would not yet exist.

## 6.6 RNN versus LSTM and clipping

A plain RNN repeatedly rewrites one hidden state with a tanh transformation, so repeated multiplication can cause gradients to vanish or explode over 128 steps. The LSTM additionally maintains a cell state:

`c_t = f_t * c_(t-1) + i_t * g_t`

The forget gate controls retention, while the input gate controls how much of each candidate is added. This additive cell update carries information across many steps more reliably than repeatedly transforming the hidden state alone. Gradient clipping guards against large or exploding parameter-gradient updates; it does not prevent all vanishing gradients, but limits the total gradient norm to 1.0.
"""
    with open("results/analysis.md", "w", encoding="utf-8") as f:
        f.write(text)


if __name__ == "__main__":
    # Running this module evaluates the checkpoint and rebuilds all reports.
    test_once()
