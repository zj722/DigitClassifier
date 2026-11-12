"""Draw the parameter-free Part 2 roadmap; leave the final shape diagram intact."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D
import numpy as np


OUT = Path(__file__).resolve().parent / "tutorial_1_cnn"
INK = "#34788d"


def main():
    with plt.rc_context({"font.family": "DejaVu Serif", "font.size": 11}):
        fig, ax = plt.subplots(figsize=(14, 3.8))
        ax.set(xlim=(-0.15, 15.3), ylim=(-0.6, 3.7), aspect="equal")
        ax.axis("off")

        def plane(x, y, width, height, shear=0.38, **kwargs):
            corners = np.array([[x, y], [x+width, y+shear],
                                [x+width, y+height+shear], [x, y+height]])
            ax.add_patch(Polygon(corners, closed=True, fill=False,
                                 edgecolor=INK, linewidth=0.85, **kwargs))
            return corners

        def stack(x, height, count, width=0.75):
            y = 1.8 - height/2
            for depth in reversed(range(count)):
                plane(x+depth*0.095, y+depth*0.045, width, height,
                      alpha=0.45 if depth else 1)
            return x, y, width, height

        digit = np.load(OUT / "digit_5.npy")
        transform = Affine2D.from_values(0.85/28, 0.38/28, 0, 2.3/28, 0, 0.65)
        ax.imshow(np.flipud(digit), origin="lower", cmap="gray_r", vmin=0, vmax=255,
                  extent=(0, 28, 0, 28), transform=transform+ax.transData,
                  interpolation="nearest", zorder=0)
        plane(0, 0.65, 0.85, 2.3)
        layers = [(0, 0.65, 0.85, 2.3), stack(2.2, 2.3, 4),
                  stack(4.4, 1.35, 4), stack(6.4, 1.35, 8), stack(8.9, 0.7, 8)]
        # A small local window connects to a response in the next map.
        for left, right in zip(layers, layers[1:]):
            x, y, width, height = left
            window = plane(x+width*0.3, y+height*0.45, 0.22, 0.28, shear=0.10)
            destination = (right[0]+0.18, right[1]+right[3]*0.53)
            for start in window:
                ax.plot([start[0], destination[0]], [start[1], destination[1]],
                        color=INK, lw=0.65, alpha=0.7)
            ax.plot(*destination, marker=".", color=INK, ms=3)

        # Flatten and linear classification: individual entries and weighted links.
        vectors = [(11.05, np.linspace(0.7, 3.1, 15)),
                   (13.5, np.linspace(0.7, 3.1, 10))]
        for x, entries in vectors:
            for y in entries:
                ax.plot([x-0.07, x+0.07, x+0.07, x-0.07, x-0.07],
                        [y-.055, y-.055, y+.055, y+.055, y-.055], color=INK, lw=.75)
        for y in (1.6, 1.8, 2.0, 2.2):
            ax.plot([9.2, 10.98], [y, 0.7+(y-1.6)*2.4/.6], color=INK, lw=.65)
        for y in vectors[0][1][::3]:
            for z in vectors[1][1][::3]:
                ax.plot([11.12, 13.43], [y, z], color=INK, lw=.55, alpha=.4)
        ax.text(14.0, 3.1, "Digit 0", va="center", color=INK)
        ax.text(14.0, 1.85, r"$\vdots$", va="center", color=INK, fontsize=18)
        ax.text(14.0, 0.7, "Digit 9", va="center", color=INK)
        # Entries run top-to-bottom from digit 0 through digit 9.
        ax.text(12.3, 3.35, "Linear", ha="center", color=INK)
        for x, label in [(0.45, "Input image"), (2.7, "Convolution\n+ ReLU"),
                         (4.9, "Pooling"), (7.05, "Convolution\n+ ReLU"),
                         (9.6, "Pooling"), (11.05, "Flatten"), (13.6, "Class scores")]:
            ax.text(x, .12, label, ha="center", va="top", color=INK)
        fig.savefig(OUT / "cnn_roadmap_wireframe.png", dpi=180, bbox_inches="tight")
        fig.savefig(OUT / "cnn_roadmap_wireframe.svg", bbox_inches="tight")
        plt.close(fig)


if __name__ == "__main__":
    main()
