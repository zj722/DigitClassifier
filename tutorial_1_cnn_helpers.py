"""Small H×W×C calculations and presentation helpers for Tutorial 1, Part 2."""
from pathlib import Path
from html import escape

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ASSETS = Path(__file__).parent / 'assets' / 'tutorial_1_cnn'

# Small demonstrations are separate from the complete 28×28 digit model.
SMALL_IMAGE = np.array([
    [3, 3, 2, 1, 0],
    [0, 0, 1, 3, 1],
    [3, 1, 2, 2, 3],
    [2, 0, 0, 2, 2],
    [2, 0, 0, 0, 1],
], dtype=float)
SMALL_KERNEL = np.array([[1, 0, -1], [1, 0, -1], [1, 0, -1]], dtype=float)
SMALL_BIAS = 1.0

CHANNEL_PATCHES = np.stack([
    [[1, 0, 2], [0, 1, 0], [1, 0, 1]],
    [[0, 1, 0], [2, 0, 1], [0, 1, 0]],
    np.ones((3, 3)),
    [[2, 0, 0], [0, 1, 0], [0, 0, 2]],
], axis=-1).astype(float)
CHANNEL_FILTER = np.stack([SMALL_KERNEL, np.ones((3, 3)), np.eye(3), -np.eye(3)], axis=-1)
CHANNEL_BIAS = 1.0
RELU_INPUT = np.array([[-2, 3, -1, 0], [4, -3, 1, 5]], dtype=float)
# Reused numerical example from tutorial_1_data_and_cnn.ipynb, cell 19.
POOL_INPUT = np.array([[1, 3, 2, 0], [4, 2, 1, 5],
                       [0, 1, 7, 2], [2, 6, 3, 4]], dtype=float)


def output_size(input_size, kernel_size, stride=1, padding=0):
    return (input_size + 2 * padding - kernel_size) // stride + 1


def convolve(image, filters, biases, stride=1, padding=0):
    """Ordinary convolution (sliding dot products), with an H×W×C input.

    filters has shape (number_of_filters, kernel_height, kernel_width, C).
    Each filter sums over all input channels and adds one bias.
    """
    image = np.asarray(image, dtype=float)
    filters = np.asarray(filters, dtype=float)
    biases = np.asarray(biases, dtype=float)
    if image.ndim != 3 or filters.ndim != 4:
        raise ValueError('Use H×W×C input and F×Kh×Kw×C filters.')
    count, kh, kw, channels = filters.shape
    if channels != image.shape[-1] or biases.shape != (count,):
        raise ValueError('Each filter must span every input channel and have one bias.')
    if stride < 1 or padding < 0:
        raise ValueError('Stride must be positive and padding nonnegative.')
    height = output_size(image.shape[0], kh, stride, padding)
    width = output_size(image.shape[1], kw, stride, padding)
    if height < 1 or width < 1:
        raise ValueError('The padded input must contain at least one complete window.')
    padded = np.pad(image, ((padding, padding), (padding, padding), (0, 0)))
    result = np.empty((height, width, count))
    for row in range(height):
        for col in range(width):
            patch = padded[row * stride:row * stride + kh,
                           col * stride:col * stride + kw, :]
            for f in range(count):
                result[row, col, f] = (patch * filters[f]).sum() + biases[f]
    return result


def max_pool2(image):
    """2×2 maximum, stride 2, separately for each H×W×C channel."""
    h, w, channels = image.shape
    result = np.empty((h // 2, w // 2, channels))
    for row in range(h // 2):
        for col in range(w // 2):
            result[row, col] = image[2*row:2*row+2, 2*col:2*col+2].max(axis=(0, 1))
    return result


def load_digit():
    """A local copy of MNIST training image 0 (label 5); brightness is 0–255."""
    return np.load(ASSETS / 'digit_5.npy').astype(float)[..., None]


def cross_entropy(scores, correct_class):
    """Stable classification loss from raw scores and a class index."""
    shifted = np.asarray(scores) - np.max(scores)
    return float(np.log(np.exp(shifted).sum()) - shifted[correct_class])


class Table:
    def __init__(self, headers, rows):
        self.headers, self.rows = headers, list(rows)

    def _repr_html_(self):
        def row(values, tag):
            return '<tr>' + ''.join(f'<{tag} style="padding:5px 12px;text-align:left">'
                                    f'{escape(str(v))}</{tag}>' for v in values) + '</tr>'
        return '<table>' + row(self.headers, 'th') + ''.join(row(r, 'td') for r in self.rows) + '</table>'

    def __repr__(self):
        return '\n'.join(' | '.join(map(str, row)) for row in [self.headers] + self.rows)


def matrix(ax, values, title='', cmap='Blues', limit=None, text_size=12):
    """Display a small numeric matrix with cell borders."""
    values = np.asarray(values)
    kwargs = {} if limit is None else {'vmin': -limit, 'vmax': limit}
    palette = plt.get_cmap(cmap).copy()
    palette.set_bad('#e2e8f0')
    ax.imshow(values, cmap=palette, **kwargs)
    for (r, c), value in np.ndenumerate(values):
        if np.isfinite(value):
            ax.text(c, r, f'{0 if value == 0 else value:g}', ha='center', va='center', fontsize=text_size,
                    color='#132b43', bbox=dict(fc='white', ec='none', alpha=0.75, pad=1))
    ax.set_xticks(np.arange(values.shape[1]+1)-0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0]+1)-0.5, minor=True)
    ax.grid(which='minor', color='white', linewidth=1.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set_title(title, fontsize=11, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)


def show_pixels(image):
    values = image[..., 0]
    # Fixed patch includes both dark background and the digit's upper stroke.
    row, col, size = 5, 10, 6
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), layout='constrained')
    axes[0].imshow(values, cmap='gray', vmin=0, vmax=255)
    axes[0].add_patch(Rectangle((col-.5, row-.5), size, size, fill=False, ec='#ec7c26', lw=2))
    axes[0].set(title='One digit image: 28 × 28 × 1', xlabel='Width: 28 columns', ylabel='Height: 28 rows')
    matrix(axes[1], values[row:row+size, col:col+size], 'Highlighted 6 × 6 patch: exact pixel values', text_size=10)
    plt.show()


def show_feature_maps(maps, title):
    channels = maps.shape[-1]
    rows = (channels + 3) // 4
    fig, axes = plt.subplots(rows, 4, figsize=(10, 2.5 * rows), squeeze=False, layout='constrained')
    limit = max(float(np.abs(maps).max()), 1e-10)
    for i, ax in enumerate(axes.flat):
        if i < channels:
            ax.imshow(maps[..., i], cmap='coolwarm', vmin=-limit, vmax=limit)
            ax.set_title(f'Channel {i}: {maps.shape[0]} × {maps.shape[1]}', fontsize=10)
        ax.axis('off')
    fig.suptitle(title + '\nSame color scale across channels; red = positive, blue = negative.', fontsize=12)
    plt.show()


def show_scores(scores, target):
    winner = int(np.argmax(scores))
    colors = ['#d56827' if digit == winner else '#7797b5' for digit in range(10)]
    fig, ax = plt.subplots(figsize=(8, 3), layout='constrained')
    ax.bar(range(10), scores, color=colors)
    ax.axhline(0, color='#4b5563', lw=0.8)
    ax.set(xticks=range(10), xlabel='Digit class', ylabel='Raw class score',
           title=f'Untrained model: predicted {winner}; correct label {target}')
    plt.show()
