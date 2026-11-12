# Tutorial 1, Part 2: visual sources

The reference notebook `../../tutorial_1_data_and_cnn.ipynb` and its referenced local assets were inspected before preparing this section. Several frames of each animation were checked, including the first, middle, and final frames.

## Reused material

- `../padding.gif`: unchanged local copy of Vincent Dumoulin and Francesco Visin's [same-padding animation](https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/same_padding_no_strides.gif). The example uses a 5×5 input, a 3×3 kernel, padding 1, and stride 1, producing a 5×5 output.
- `../stride.gif`: unchanged local copy of their [stride animation](https://github.com/vdumoulin/conv_arithmetic/blob/master/gif/no_padding_strides.gif). The example uses a 5×5 input, a 3×3 kernel, padding 0, and stride 2, producing a 2×2 output.
- Both existing files were checked byte-for-byte against the upstream originals. The upstream [README](https://github.com/vdumoulin/conv_arithmetic) explicitly permits reuse of the code and images with attribution under its MIT license. The complete license is included in `conv_arithmetic_LICENSE.txt`. Original work: [A guide to convolution arithmetic for deep learning](https://arxiv.org/abs/1603.07285), Vincent Dumoulin and Francesco Visin (2016). Blue maps are inputs; cyan maps are outputs.
- The 4×4 numerical pooling input is reused from cell 19 of `tutorial_1_data_and_cnn.ipynb`. Its new diagram computes and highlights the maxima.
- `digit_5.npy` contains only training image 0, label 5, copied from the repository's existing MNIST IDX cache. Dataset authors: Yann LeCun, Corinna Cortes, and Christopher J. C. Burges; [dataset and distribution information](https://github.com/cvdfoundation/mnist). It is used to demonstrate a forward computation, with no train/test evaluation. The notebook reads this small local sample and downloads nothing.

## Exact illustrations created for this model

`../generate_tutorial_1_cnn_figures.py` generates the following original figures using NumPy and Matplotlib. Numerical inputs and computations are shared with `../../tutorial_1_cnn_helpers.py`:

| File | Purpose |
|---|---|
| `sliding_window.gif`, `window_first.png` | Synchronized input window, fixed weights, products, and corresponding output; 5×5 single-channel demonstration, bias 1 |
| `channel_sum.png` | Four channel-wise weighted contributions summed, with one bias, into one output value |
| `spatial_sizes.png` | Separate changes to kernel size, stride, and padding on a 5×5 demonstration |
| `relu_values.png` | Exact before/after values for ReLU |
| `max_pool.png` | Selected maximum and corresponding output for each 2×2 window |
| `layer_connections.png` | Four Conv 1 maps retained through ReLU/pooling, then used together by every one of Conv 2's eight filters |
| `complete_model.png` | The requested two-block CNN, all output shapes, and learnable layers |

The reference's other convolution GIFs and its ReLU image were inspected. The arithmetic GIF uses different weights and a zero bias; its channel examples use three color channels. Those illustrations do not provide the synchronized products or exact four-to-eight-channel connections required here. The ReLU image has no identified original creator in the supplied provenance. Online reference material, including the [original convolution-arithmetic collection](https://github.com/vdumoulin/conv_arithmetic) and [CS231n's convolution notes](https://cs231n.github.io/convolutional-networks/), was also reviewed. Precise new diagrams were used for the requested model and exact numerical examples; unrelated architectures were not inserted.

The revised Section 2.2.1 computes three illustrative feature maps inline in PyTorch using **hand-set** vertical-edge, horizontal-edge, and averaging filters. Edge panels display absolute responses, and each panel has its own brightness scale. The complete model and its class scores use seeded, **untrained** weights; neither example represents learned digit detectors.

## Revised Part 2 illustrations

`cnn_roadmap_wireframe.png` (also supplied as SVG) is an original Matplotlib drawing inspired by the thin-line, feature-plane style of the reference image supplied by the user. It uses the local digit rather than copying the reference's vehicle artwork. It is a schematic module overview; the final `complete_model.png` retains the exact shape summary in the unchanged Section 2.6.

The pixel patch, hand-set feature maps, and three-panel function-composition plot are generated directly in the notebook. The two-channel example explicitly pairs every input entry with a filter weight; its PyTorch output verifies the two separate responses, 17 and 16. Existing sliding-window, spatial-size, and pooling illustrations are retained.

To regenerate the new roadmap without changing the existing figures:

```sh
python number_classifier/assets/generate_tutorial_1_wireframe.py
```

PyTorch references: [Conv2d and weight layout](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html), [ReLU](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html), and [MaxPool2d](https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html). The notebook's forward pass is checked using PyTorch 2.5.1.

To regenerate the static figures from the repository root:

```sh
python number_classifier/assets/generate_tutorial_1_cnn_figures.py
```
