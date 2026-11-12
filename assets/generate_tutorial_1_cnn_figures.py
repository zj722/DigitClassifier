"""Regenerate the precise illustrations for Tutorial 1, Part 2.

Run with Python from any working directory. Uses the included local digit sample.
"""
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import tutorial_1_cnn_helpers as cnn

OUT = cnn.ASSETS
OUT.mkdir(exist_ok=True)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})


def save(fig, name):
    fig.savefig(OUT / name, dpi=140, bbox_inches='tight')
    plt.close(fig)


def sliding_window():
    inputs, kernel, bias = cnn.SMALL_IMAGE, cnn.SMALL_KERNEL, cnn.SMALL_BIAS
    outputs = cnn.convolve(inputs[..., None], kernel[None, ..., None], np.array([bias]))[..., 0]
    fig, axes = plt.subplots(1, 4, figsize=(12.5, 3.8))
    fig.subplots_adjust(left=0.025, right=0.985, top=0.80, bottom=0.23, wspace=0.32)
    equation = fig.text(0.5, 0.08, '', ha='center', fontsize=13)
    fig.suptitle('Small demonstration: move one window, write one output value', fontsize=14)

    def draw(step):
        row, col = divmod(step, outputs.shape[1])
        for ax in axes:
            ax.clear()
        patch = inputs[row:row+3, col:col+3]
        products = patch * kernel
        visible = np.full_like(outputs, np.nan)
        visible.flat[:step+1] = outputs.flat[:step+1]
        cnn.matrix(axes[0], inputs, f'Input 5 × 5; window ({row}, {col})', text_size=11)
        axes[0].add_patch(Rectangle((col-.5, row-.5), 3, 3, fill=False, ec='#e27021', lw=3))
        cnn.matrix(axes[1], kernel, 'Same 3 × 3 weights at every step', 'coolwarm', limit=1)
        cnn.matrix(axes[2], products, 'Multiply corresponding entries', 'coolwarm', limit=3)
        cnn.matrix(axes[3], visible, 'Output: write the highlighted cell', 'coolwarm', limit=7)
        axes[3].add_patch(Rectangle((col-.5, row-.5), 1, 1, fill=False, ec='#e27021', lw=3))
        equation.set_text(f'Sum of products = {products.sum():g}     +     one bias = {bias:g}'
                          f'     →     output[{row}, {col}] = {outputs[row,col]:g}')
        return []

    draw(0)
    fig.savefig(OUT / 'window_first.png', dpi=140)
    animation = FuncAnimation(fig, draw, frames=9, interval=1200, repeat=True)
    animation.save(OUT / 'sliding_window.gif', writer=PillowWriter(fps=0.8), dpi=110)
    plt.close(fig)


def channel_sum():
    contributions = (cnn.CHANNEL_PATCHES * cnn.CHANNEL_FILTER).sum(axis=(0, 1))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    fig.subplots_adjust(top=0.84, bottom=0.16, hspace=0.65, wspace=0.38)
    for c in range(4):
        row, pair = divmod(c, 2)
        cnn.matrix(axes[row, pair*2], cnn.CHANNEL_PATCHES[..., c], f'Input channel {c}: one patch')
        cnn.matrix(axes[row, pair*2+1], cnn.CHANNEL_FILTER[..., c],
                   f'Weight slice {c}\nSum of products = {contributions[c]:g}', 'coolwarm', limit=1)
    fig.suptitle('ONE filter has FOUR weight slices: one for each input channel', fontsize=14)
    total = contributions.sum() + cnn.CHANNEL_BIAS
    terms = ' + '.join(f'({v:g})' if v < 0 else f'{v:g}' for v in contributions)
    fig.text(0.5, 0.055, f'Add channel contributions: {terms}    +    one bias {cnn.CHANNEL_BIAS:g}'
             f'    =    ONE output value {total:g}', ha='center', fontsize=12)
    save(fig, 'channel_sum.png')


def spatial_sizes():
    configs = [(1, 1, 0), (3, 1, 0), (3, 2, 0), (3, 1, 1)]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.8), layout='constrained')
    for ax, (kernel, stride, padding) in zip(axes, configs):
        grid = np.pad(np.ones((5, 5)), padding)
        out = cnn.output_size(5, kernel, stride, padding)
        cnn.matrix(ax, grid, f'K={kernel}, S={stride}, P={padding}\nOutput {out} × {out}', text_size=8)
        ax.add_patch(Rectangle((-.5, -.5), kernel, kernel, fill=False, ec='#12649a', lw=2.5))
        ax.add_patch(Rectangle((stride-.5, -.5), kernel, kernel, fill=False, ec='#dc6923', lw=2.5, ls='--'))
        if padding:
            ax.add_patch(Rectangle((padding-.5, padding-.5), 5, 5, fill=False, ec='#27364b', lw=1))
        ax.set_xlabel(f'{out} valid starts per axis', fontsize=10)
    fig.suptitle('Same 5 × 5 input: first window (blue), next window (orange); border zeros are padding', fontsize=12)
    save(fig, 'spatial_sizes.png')


def relu_and_pool():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 2.8), layout='constrained')
    cnn.matrix(axes[0], cnn.RELU_INPUT, 'Before: signed responses', 'coolwarm', limit=5)
    cnn.matrix(axes[1], np.maximum(cnn.RELU_INPUT, 0), 'After: negative values become 0', 'coolwarm', limit=5)
    save(fig, 'relu_values.png')

    result = cnn.max_pool2(cnn.POOL_INPUT[..., None])[..., 0]
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), layout='constrained')
    cnn.matrix(axes[0], cnn.POOL_INPUT, '4 × 4 input: four separate 2 × 2 windows')
    cnn.matrix(axes[1], result, '2 × 2 output: one maximum per window')
    colors = ['#c65b28', '#18769b', '#3f8a4b', '#9653a3']
    for i, color in enumerate(colors):
        r, c = divmod(i, 2)
        window = cnn.POOL_INPUT[r*2:r*2+2, c*2:c*2+2]
        mr, mc = np.unravel_index(np.argmax(window), window.shape)
        axes[0].add_patch(Rectangle((c*2-.5, r*2-.5), 2, 2, fill=False, ec=color, lw=2.5))
        axes[0].add_patch(Rectangle((c*2+mc-.43, r*2+mr-.43), .86, .86, fill=False, ec=color, lw=3))
        axes[1].add_patch(Rectangle((c-.43, r-.43), .86, .86, fill=False, ec=color, lw=3))
    save(fig, 'max_pool.png')


def connections():
    fig, ax = plt.subplots(figsize=(13, 6.2))
    ax.set(xlim=(0, 14), ylim=(0, 6.7))
    ax.axis('off')
    def box(x, y, w, h, text, color='#eaf2f9', fontsize=10):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02',
                     fc=color, ec='#65758b', lw=1))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=fontsize)
    def arrow(start, end):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='#65758b', lw=1))
    ax.text(1.15, 6.25, 'Conv 1 outputs\n28 × 28 × 4', ha='center', fontsize=12)
    ax.text(5.1, 6.25, 'Inputs to Conv 2\n14 × 14 × 4', ha='center', fontsize=12)
    ax.text(9.1, 6.25, '8 different filters\neach 3 × 3 × 4', ha='center', fontsize=12)
    ax.text(12.55, 6.25, 'Conv 2 outputs\n14 × 14 × 8', ha='center', fontsize=12)
    colors = ['#d9eafa', '#dff0e3', '#fff0d8', '#ede2f6']
    centers = np.linspace(4.9, 1.3, 4)
    box(2.6, 1.0, 1.6, 4.2, 'ReLU\n+\n2 × 2\nMaxPool', '#f1f3f5')
    for c, (y, color) in enumerate(zip(centers, colors)):
        box(.3, y-.35, 1.7, .7, f'Map {c}', color)
        arrow((2.0, y), (2.58, y))
        box(4.55, y-.35, 1.1, .7, f'Map {c}', color)
        arrow((4.22, y), (4.53, y))
        ax.plot([5.67, 6.8], [y, y], color='#65758b', lw=1)
    ax.plot([6.8, 6.8], [.6, 5.25], color='#65758b', lw=1.5)
    ax.text(6.8, 5.6, 'All 4 maps\nto every filter', ha='center', fontsize=10)
    for f, y in enumerate(np.linspace(5.1, .7, 8)):
        arrow((6.8, y), (8.15, y))
        box(8.18, y-.22, 1.85, .44, f'Filter {f}', '#fff0de', 10)
        arrow((10.05, y), (11.68, y))
        box(11.7, y-.22, 1.7, .44, f'Map {f}', '#e3f1eb', 10)
    ax.text(7, .05, 'Each filter sums across its 4 input channels internally. The 8 output maps are kept as separate channels.',
            ha='center', fontsize=10)
    save(fig, 'layer_connections.png')


def roadmap():
    fig, ax = plt.subplots(figsize=(13.5, 5.5))
    ax.set(xlim=(0, 14.4), ylim=(0, 5.7))
    ax.axis('off')
    top = [
        ('Input', 'grayscale brightness', '28 × 28 × 1', False),
        ('Conv 1', '4 filters; 3 × 3; S=1, P=1', '28 × 28 × 4', True),
        ('ReLU', 'max(0, value)', '28 × 28 × 4', False),
        ('MaxPool', '2 × 2; stride 2', '14 × 14 × 4', False),
    ]
    bottom = [
        ('Conv 2', '8 filters; 3 × 3; S=1, P=1', '14 × 14 × 8', True),
        ('ReLU', 'max(0, value)', '14 × 14 × 8', False),
        ('MaxPool', '2 × 2; stride 2', '7 × 7 × 8', False),
        ('Flatten', 'keep all entries', '392 values', False),
        ('Linear', 'one score per digit', '10 class scores', True),
    ]
    def card(x, y, width, item):
        title, desc, shape, learned = item
        ax.add_patch(FancyBboxPatch((x, y), width, 1.48, boxstyle='round,pad=0.03',
                     fc='#fff0df' if learned else '#edf3f8', ec='#c67935' if learned else '#657b92'))
        ax.text(x+width/2, y+1.19, title, ha='center', fontsize=12, fontweight='bold')
        ax.text(x+width/2, y+.85, desc, ha='center', fontsize=8.5)
        ax.text(x+width/2, y+.48, shape, ha='center', fontsize=12)
        ax.text(x+width/2, y+.15, 'learnable weights + biases' if learned else 'no learnable parameters',
                ha='center', fontsize=8, color='#95501d' if learned else '#53657a')
    def arrow(start, end):
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', color='#526578', lw=1.6))
    top_x = [.25, 3.75, 7.25, 10.75]
    bot_x = [.25, 3.05, 5.85, 8.65, 11.45]
    for x, item in zip(top_x, top):
        card(x, 3.65, 3.05, item)
    for x, item in zip(bot_x, bottom):
        card(x, .35, 2.45, item)
    for i in range(3):
        arrow((top_x[i]+3.08, 4.39), (top_x[i+1]-.04, 4.39))
    for i in range(4):
        arrow((bot_x[i]+2.48, 1.09), (bot_x[i+1]-.04, 1.09))
    ax.plot([12.28, 12.28, 1.48], [3.61, 2.7, 2.7], color='#526578', lw=1.6)
    arrow((1.48, 2.7), (1.48, 1.88))
    ax.text(7.2, 5.48, 'Our complete CNN • output shapes use H × W × C • orange layers have learnable parameters',
            ha='center', fontsize=12)
    save(fig, 'complete_model.png')


if __name__ == '__main__':
    sliding_window()
    channel_sum()
    spatial_sizes()
    relu_and_pool()
    connections()
    roadmap()
    print(f'Generated Part 2 figures in {OUT}')
