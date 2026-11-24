"""Generate a LinkedIn-sized poster from the verified HAR experiment results."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path("results/linkedin_poster.png")
NAVY = "#0B1220"
PANEL = "#121D30"
WHITE = "#F4F7FB"
MUTED = "#A6B4C7"
TEAL = "#42E2C2"
BLUE = "#5B9BFF"
BAR_BG = "#26354A"


def rounded_box(ax, xy, width, height, color, radius=0.018, edge=None, lw=1):
    """Draw a rounded rectangle in poster-relative coordinates."""
    patch = FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle=f"round,pad=0.005,rounding_size={radius}",
        facecolor=color,
        edgecolor=edge or color,
        linewidth=lw,
        transform=ax.transAxes,
    )
    ax.add_patch(patch)


def build_poster():
    """Render the project summary and measured test results to a PNG."""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.8, 13.5), dpi=100, facecolor=NAVY)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Header identifies the project and gives the poster a short hook.
    rounded_box(ax, (0.075, 0.925), 0.34, 0.035, PANEL, radius=0.012)
    ax.text(
        0.095,
        0.942,
        "APPLIED MACHINE LEARNING  /  UCI HAR",
        color=TEAL,
        fontsize=10,
        fontweight="bold",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.075,
        0.875,
        "Human Activity\nRecognition",
        color=WHITE,
        fontsize=35,
        fontweight="bold",
        linespacing=1.05,
        va="top",
        transform=ax.transAxes,
    )
    ax.text(
        0.078,
        0.785,
        "Can a phone understand how you move?",
        color=MUTED,
        fontsize=16,
        va="top",
        transform=ax.transAxes,
    )

    # Highlight the final score and the size of the held-out test set.
    rounded_box(ax, (0.075, 0.625), 0.85, 0.12, PANEL, radius=0.025)
    ax.text(
        0.115,
        0.685,
        "91.21%",
        color=TEAL,
        fontsize=43,
        fontweight="bold",
        va="center",
        transform=ax.transAxes,
    )
    ax.plot(
        [0.57, 0.57],
        [0.65, 0.72],
        color="#34445C",
        linewidth=1.3,
        transform=ax.transAxes,
        clip_on=False,
    )
    ax.text(
        0.61,
        0.696,
        "FINAL TEST ACCURACY",
        color=WHITE,
        fontsize=12,
        fontweight="bold",
        va="center",
        transform=ax.transAxes,
    )
    ax.text(
        0.61,
        0.66,
        "2,947 held-out test windows",
        color=MUTED,
        fontsize=12,
        va="center",
        transform=ax.transAxes,
    )

    # Summarize the input window and the main model stages.
    ax.text(
        0.08,
        0.585,
        "FROM RAW MOTION TO ACTIVITY",
        color=WHITE,
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.08,
        0.553,
        "9 sensor channels  |  128 samples  |  2.56 seconds",
        color=MUTED,
        fontsize=11,
        transform=ax.transAxes,
    )

    blocks = [
        (0.08, 0.475, 0.23, "RAW SENSOR\nWINDOW"),
        (0.385, 0.475, 0.23, "BIDIRECTIONAL\nLSTM"),
        (0.69, 0.475, 0.23, "ATTENTION\n+ 6 CLASSES"),
    ]
    for x, y, w, label in blocks:
        rounded_box(ax, (x, y), w, 0.06, PANEL, radius=0.015, edge="#2B3D55", lw=1.2)
        ax.text(
            x + w / 2,
            y + 0.03,
            label,
            color=WHITE,
            fontsize=11,
            fontweight="bold",
            ha="center",
            va="center",
            linespacing=1.15,
            transform=ax.transAxes,
        )
    for x in (0.325, 0.63):
        ax.annotate(
            "",
            xy=(x + 0.045, 0.505),
            xytext=(x, 0.505),
            xycoords=ax.transAxes,
            textcoords=ax.transAxes,
            arrowprops={"arrowstyle": "->", "color": TEAL, "lw": 1.8},
        )

    # Compare class-level scores; blue marks the two weaker classes.
    ax.text(
        0.08,
        0.425,
        "PER-ACTIVITY ACCURACY",
        color=WHITE,
        fontsize=12,
        fontweight="bold",
        transform=ax.transAxes,
    )
    classes = [
        ("Walking", 0.9536),
        ("Upstairs", 0.9682),
        ("Downstairs", 0.9643),
        ("Sitting", 0.8452),
        ("Standing", 0.8064),
        ("Laying", 0.9497),
    ]
    label_x, bar_x, bar_w, value_x = 0.09, 0.34, 0.43, 0.90
    top_y, row_gap = 0.382, 0.038
    for idx, (name, value) in enumerate(classes):
        y = top_y - idx * row_gap
        ax.text(
            label_x,
            y,
            name,
            color=WHITE,
            fontsize=11,
            va="center",
            transform=ax.transAxes,
        )
        rounded_box(ax, (bar_x, y - 0.008), bar_w, 0.016, BAR_BG, radius=0.007)
        fill_color = TEAL if value >= 0.90 else BLUE
        rounded_box(
            ax, (bar_x, y - 0.008), bar_w * value, 0.016, fill_color, radius=0.007
        )
        ax.text(
            value_x,
            y,
            f"{value * 100:.1f}%",
            color=WHITE,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="right",
            transform=ax.transAxes,
        )

    # Finish with the main confusion, author credit, and shareable tags.
    rounded_box(ax, (0.075, 0.073), 0.85, 0.075, PANEL, radius=0.018)
    ax.text(
        0.105,
        0.119,
        "THE BIGGEST CHALLENGE",
        color=TEAL,
        fontsize=10,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.105,
        0.091,
        "Sitting vs. standing: similar low-motion sensor patterns.",
        color=WHITE,
        fontsize=11,
        transform=ax.transAxes,
    )
    ax.text(
        0.08,
        0.052,
        "Created by Muhammad Mubeen",
        color=WHITE,
        fontsize=9.5,
        fontweight="bold",
        transform=ax.transAxes,
    )
    ax.text(
        0.08,
        0.033,
        "PyTorch  |  Manual LSTM + attention  |  Subject-independent validation",
        color=MUTED,
        fontsize=9.5,
        transform=ax.transAxes,
    )
    ax.text(
        0.08,
        0.014,
        "#MachineLearning  #PyTorch  #HumanActivityRecognition",
        color=TEAL,
        fontsize=9.5,
        fontweight="bold",
        transform=ax.transAxes,
    )

    fig.savefig(OUT, dpi=100, facecolor=NAVY)
    plt.close(fig)
    print(f"Saved LinkedIn poster: {OUT.resolve()}")


if __name__ == "__main__":
    # Regenerate the poster when this script is executed directly.
    build_poster()
