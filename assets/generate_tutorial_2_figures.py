"""Generate original teaching diagrams for Tutorial 2 (no training results)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "tutorial_2"
OUT.mkdir(exist_ok=True)
BLUE, ORANGE, GREEN = "#2563a6", "#c06424", "#237a61"
INK = "#183047"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 12})


def box(ax, xy, width, height, title, body, color):
    x, y = xy
    ax.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.015,rounding_size=0.025",
        facecolor=color + "12", edgecolor=color, linewidth=1.5))
    ax.text(x + width / 2, y + height * .72, title, ha="center", va="center",
            color=color, fontsize=14, weight="bold")
    ax.text(x + width / 2, y + height * .31, body, ha="center", va="center",
            color=INK, fontsize=11, linespacing=1.5)


def arrow(ax, start, end, color=INK):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
                                mutation_scale=16, color=color, linewidth=1.5))


def save(fig, name):
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)


fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
ax.set_title("One fixed split; three different jobs", fontsize=19, pad=18, color=INK)
box(ax, (.04, .70), .56, .21, "Official training portion: 60,000",
    "Shuffle sample indices once, using seed 42", BLUE)
box(ax, (.04, .20), .25, .28, "Train: 50,000",
    "Images + labels\nUpdate model parameters", BLUE)
box(ax, (.35, .20), .25, .28, "Validate: 10,000",
    "Separate images + labels\nChoose settings / checkpoint", ORANGE)
box(ax, (.70, .20), .25, .28, "Test: 10,000",
    "Official test portion\nFinal project evaluation", GREEN)
arrow(ax, (.18, .69), (.165, .49), BLUE)
arrow(ax, (.46, .69), (.475, .49), ORANGE)
ax.text(.825, .75, "Keep reserved", ha="center", color=GREEN,
        fontsize=15, weight="bold")
ax.text(.825, .61, "Never used to select\nthis tutorial's model", ha="center", color=INK)
ax.text(.50, .055, "Split membership stays fixed. Training batch order changes each epoch.",
        ha="center", color=INK, fontsize=12)
save(fig, "data_split.png")

fig, ax = plt.subplots(figsize=(12, 5.5))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
ax.set_title("One epoch: learn, measure, then decide", fontsize=19, pad=18, color=INK)
box(ax, (.03, .60), .40, .26, "1. Train on shuffled batches",
    "zero_grad → forward → loss\nbackward → optimizer.step", BLUE)
box(ax, (.57, .60), .40, .26, "2. Evaluate the current weights",
    "model.eval() + torch.no_grad()\nTrain set and validation set; no updates", ORANGE)
arrow(ax, (.44, .73), (.56, .73))
box(ax, (.57, .13), .40, .25, "3. Record and select",
    "Log loss + accuracy\nSave the lowest-validation-loss weights", GREEN)
arrow(ax, (.77, .59), (.77, .39))
box(ax, (.03, .13), .40, .25, "4. Continue or stop",
    "Next epoch, or stop after patience\nRestore the best saved weights", INK)
arrow(ax, (.56, .25), (.44, .25))
arrow(ax, (.23, .39), (.23, .59))
save(fig, "epoch_workflow.png")

fig, ax = plt.subplots(figsize=(12, 5.6))
ax.set(xlim=(0, 1), ylim=(0, 1))
ax.axis("off")
ax.set_title("Two CNNs, the same ten-class task", fontsize=19, pad=18, color=INK)
for y, label, color, channels, head in [
    (.58, "A · one convolution", BLUE, "4 × 14 × 14", "784 → 10"),
    (.12, "B · two convolutions", ORANGE, "16 × 14 × 14", "1,568 → 10"),
]:
    ax.text(.025, y + .26, label, color=color, fontsize=14, weight="bold")
    box(ax, (.025, y), .14, .20, "Input", "1 × 28 × 28", color)
    box(ax, (.215, y), .23, .20, "Conv → ReLU → Pool", channels, color)
    arrow(ax, (.17, y + .10), (.21, y + .10), color)
    if label.startswith("B"):
        box(ax, (.495, y), .23, .20, "Conv → ReLU → Pool", "32 × 7 × 7", color)
        arrow(ax, (.45, y + .10), (.49, y + .10), color)
        arrow(ax, (.73, y + .10), (.78, y + .10), color)
    else:
        arrow(ax, (.45, y + .10), (.78, y + .10), color)
        ax.text(.61, y + .14, "flatten", ha="center", color=INK)
    box(ax, (.785, y), .19, .20, "Linear → logits", head, color)
ax.text(.50, .025, "Every convolution: 3 × 3, stride 1, padding 1. Every pool: 2 × 2, stride 2.",
        ha="center", color=INK, fontsize=11)
save(fig, "cnn_comparison.png")
print(f"Saved three original diagrams to {OUT}")
