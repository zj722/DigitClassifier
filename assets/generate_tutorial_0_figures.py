from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle, FancyBboxPatch
from matplotlib.path import Path as PolygonPath
import numpy as np

out = Path(__file__).resolve().parent
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 13})
ink, blue, orange = '#22354a', '#2563a6', '#b85b17'

# The pixel example uses exactly the values in Section 4.
values = np.array([[0, 128, 255], [255, 128, 0]])
fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.0))
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_xticks(range(3), ['column 0', 'column 1', 'column 2'])
    ax.set_yticks(range(2), ['row 0', 'row 1'])
    ax.tick_params(length=0, pad=8, labelsize=11, colors=ink)
    ax.xaxis.tick_top()
    ax.set_xlim(-.5, 2.5)
    ax.set_ylim(1.5, -.5)
    ax.set_aspect('equal')
    for spine in ax.spines.values(): spine.set_visible(False)
axes[0].imshow(values, cmap='gray', vmin=0, vmax=255, interpolation='nearest')
axes[0].set_title('What the image looks like', pad=42, color=ink, fontsize=14)
axes[1].set_facecolor('#f1f5f9')
axes[1].set_title('The numbers stored for it', pad=42, color=ink, fontsize=14)
for row in range(2):
    for col in range(3):
        axes[1].add_patch(Rectangle((col-.5, row-.5), 1, 1,
                                   facecolor='#f1f5f9', edgecolor='white', linewidth=3))
        axes[1].text(col, row, str(values[row, col]), ha='center', va='center',
                     color=ink, fontsize=23)
for ax in axes:
    ax.add_patch(Rectangle((.5, -.5), 1, 1, fill=False, edgecolor=orange, linewidth=3))
fig.text(.5, .07, 'Same highlighted pixel: row 0, column 1 has brightness 128 (gray).',
         ha='center', color=ink, fontsize=12)
fig.subplots_adjust(left=.09, right=.98, bottom=.19, top=.71, wspace=.58)
fig.savefig(out/'tutorial_0_pixels.png', dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.2)
plt.close(fig)

# Diagram the actual 784-to-2 layer without placing drawing code in the lesson.
axis = np.linspace(-1.4, 1.4, 28)
x, y = np.meshgrid(axis, axis[::-1])
angles = np.pi/2 + np.arange(10)*np.pi/5
radii = np.where(np.arange(10)%2 == 0, 1., .381966)
verts = np.column_stack((radii*np.cos(angles), radii*np.sin(angles)))
star = PolygonPath(verts).contains_points(np.column_stack((x.ravel(), y.ravel()))).reshape(28,28)
fig, ax = plt.subplots(figsize=(10, 4.5))
fig.patch.set_facecolor('white')
ax.set_xlim(0, 10)
ax.set_ylim(0, 4.5)
ax.set_aspect('equal')
ax.axis('off')
ax.text(1.1, 4.13, 'One input image', ha='center', color=ink, fontsize=14, weight='bold')
ax.imshow(~star, cmap='gray', vmin=0, vmax=1, extent=(.35,1.85,1.65,3.15), interpolation='nearest')
ax.text(1.1, 1.28, '1 channel\n28 × 28 pixels', ha='center', va='center', color=ink, fontsize=12)
ax.text(3.25, 3.32, 'Flatten', ha='center', color=ink, fontsize=13)
ax.add_patch(FancyArrowPatch((2.05,2.4),(4.15,2.4),arrowstyle='-|>',mutation_scale=18,color=ink,linewidth=1.5))
ax.text(4.65, 4.13, '784 pixel inputs', ha='center', color=ink, fontsize=14, weight='bold')
inputs=[(4.65,3.35,'x0'),(4.65,2.55,'x1'),(4.65,1.05,'x783')]
outputs=[(7.7,3.15,blue),(7.7,1.45,orange)]
for ix,iy,label in inputs:
    for ox,oy,color in outputs:
        ax.add_patch(FancyArrowPatch((ix+.28,iy),(ox-.36,oy),arrowstyle='-|>',
                      mutation_scale=12,color=color,alpha=.6,linewidth=1.3))
for ix,iy,label in inputs:
    ax.add_patch(Circle((ix,iy),.3,facecolor='#edf3f8',edgecolor=ink,linewidth=1.2))
    ax.text(ix,iy,label,ha='center',va='center',color=ink,fontsize=11)
ax.text(4.65,1.8,'⋮',ha='center',va='center',color=ink,fontsize=26)
ax.text(7.7,4.13,'Linear(784, 2)',ha='center',color=ink,fontsize=14,weight='bold')
for (ox,oy,color),name,idx in zip(outputs,['Star','Heart'],[0,1]):
    ax.add_patch(Circle((ox,oy),.35,facecolor='white',edgecolor=color,linewidth=2))
    ax.text(ox,oy,'Σ',ha='center',va='center',color=color,fontsize=20)
    ax.text(ox,oy-.59,'+ own bias',ha='center',color=color,fontsize=11)
    ax.add_patch(FancyArrowPatch((ox+.39,oy),(8.5,oy),arrowstyle='-|>',
                  mutation_scale=13,color=color,linewidth=1.5))
    ax.text(8.65,oy,f'{name} score\nclass {idx}',va='center',color=color,fontsize=13)
ax.text(5.7,.28,'Each output uses all 784 inputs, with its own weights and bias.',
         ha='center',color=ink,fontsize=12)
fig.tight_layout(pad=.8)
fig.savefig(out/'tutorial_0_linear_layer.png',dpi=180,facecolor='white',bbox_inches='tight',pad_inches=.15)
plt.close(fig)
print('Created pixel and linear-layer figures.')

# Additional static teaching diagrams. These helpers run only when regenerating
# the assets; learners do not need plotting code to follow the notebook.
def canvas(width=12, height=4):
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor('white')
    ax.set(xlim=(0, width), ylim=(0, height))
    ax.set_aspect('equal')
    ax.axis('off')
    return fig, ax


def label(ax, x, y, text, size=13, color=ink, weight='normal', ha='center'):
    ax.text(x, y, text, ha=ha, va='center', color=color, fontsize=size, weight=weight)


def box(ax, x, y, w, h, text='', color=blue, fill='#f1f6fb', size=12):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04,rounding_size=0.12',
                               facecolor=fill, edgecolor=color, linewidth=1.4))
    if text:
        label(ax, x+w/2, y+h/2, text, size=size, color=color)


def arrow(ax, start, end, color=ink, style='-|>', connectionstyle='arc3'):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=16,
                                color=color, linewidth=1.5, connectionstyle=connectionstyle))


def grid(ax, x, y, values, cell=.62, colors=None, outline=ink, fontsize=13):
    """Draw values from a bottom-left origin, with row zero at the top."""
    rows, cols = len(values), len(values[0])
    for r in range(rows):
        for c in range(cols):
            face = colors[r][c] if colors else '#edf3f8'
            ax.add_patch(Rectangle((x+c*cell, y+(rows-r-1)*cell), cell, cell,
                                   facecolor=face, edgecolor='white', linewidth=2))
            label(ax, x+(c+.5)*cell, y+(rows-r-.5)*cell, str(values[r][c]), size=fontsize)
    ax.add_patch(Rectangle((x, y), cols*cell, rows*cell, fill=False, edgecolor=outline, linewidth=1.3))


def finish(fig, name):
    fig.tight_layout(pad=.4)
    fig.savefig(out/name, dpi=180, facecolor='white', bbox_inches='tight', pad_inches=.12)
    plt.close(fig)


# 1. Adding a singleton channel and then a singleton batch does not alter values.
fig, ax = canvas(12, 3.9)
positions = [.6, 4.7, 8.8]
titles = ['Pixels', 'One grayscale image', 'A batch of one image']
shapes = ['[2, 3]', '[1, 2, 3]', '[1, 1, 2, 3]']
for k, (left, title, shape) in enumerate(zip(positions, titles, shapes)):
    label(ax, left+.95, 3.48, title, size=14, weight='bold')
    if k >= 2:
        box(ax, left-.23, 1.04, 2.36, 1.82, color=orange, fill='#fff7ef')
    if k >= 1:
        box(ax, left-.12, 1.15, 2.13, 1.6, color=blue, fill='white')
    grid(ax, left, 1.35, [['0', '0.5', '1'], ['1', '0.5', '0']])
    label(ax, left+.95, .7, shape, size=15, weight='bold')
    label(ax, left+.95, .31, ['H, W', 'C = 1', 'B = 1, C = 1'][k], size=12,
          color=[ink, blue, orange][k])
for left in [2.95, 7.05]:
    arrow(ax, (left,2.0), (left+1.4,2.0))
    label(ax, left+.7, 2.5, 'unsqueeze(0)', size=12)
finish(fig, 'tutorial_0_unsqueeze.png')

# 2. Keep image membership visible while merging the other axes.
fig, ax = canvas(12, 4.3)
label(ax, 2.1, 3.88, 'Two images in a batch', size=15, weight='bold')
label(ax, 9.1, 3.88, 'One row per image', size=15, weight='bold')
first = [['0', '0.5', '1'], ['1', '0.5', '0']]
second = [['1', '0.5', '0'], ['0', '0.5', '1']]
for vals, yy, color, fill, image_id in [(first, 2.1, blue, '#edf3f8', 0),
                                       (second, .4, orange, '#fff1e6', 1)]:
    grid(ax, 1.45, yy, vals, colors=[[fill]*3]*2, outline=color)
    label(ax, .8, yy+.62, f'Image {image_id}', size=12, color=color)
    row_y = 2.7 if image_id == 0 else 1.25
    grid(ax, 7.15, row_y, [[v for row in vals for v in row]], cell=.62,
         colors=[[fill]*6], outline=color)
    arrow(ax, (3.65, yy+.62), (6.8, row_y+.31), color=color)
label(ax, 5.2, 3.36, 'flatten(1)', size=14, weight='bold')
label(ax, 5.2, 2.02, 'Read each image\nrow by row', size=12)
label(ax, 2.4, .13, '[B=2, C=1, H=2, W=3]', size=12)
label(ax, 9.0, .35, '[B=2, 6]', size=13)
finish(fig, 'tutorial_0_flatten.png')

# 3. Alias/view relationships point at shared data; a clone has separate data.
fig, ax = canvas(12, 4.2)
label(ax, 3.0, 3.77, 'Shared data', size=15, weight='bold', color=blue)
label(ax, 9.1, 3.77, 'Independent copy', size=15, weight='bold', color=orange)
box(ax, .35, 2.36, 2.15, .6, 'pixels', color=blue)
box(ax, .35, 1.16, 2.15, .6, 'first_row', color=blue)
grid(ax, 3.55, 1.6, [['0.25', '0.502', '1'], ['1', '0.502', '0']], cell=.68, fontsize=11)
ax.add_patch(Rectangle((3.55, 2.28), .68, .68, fill=False, edgecolor=orange, linewidth=3))
arrow(ax, (2.6,2.66), (3.4,2.5), color=blue)
arrow(ax, (2.6,1.46), (3.7,2.28), color=blue)
box(ax, 7.15, 3.08, 3.4, .49, 'saved_pixels = pixels.clone()', color=orange, fill='#fff7ef', size=11)
grid(ax, 8.1, 1.6, [['0', '0.502', '1'], ['1', '0.502', '0']], cell=.68, fontsize=11,
     colors=[['#fff1e6']*3]*2, outline=orange)
label(ax, 3.0, .63, 'Edit first_row[0] = 0.25', size=13, color=blue, weight='bold')
label(ax, 3.0, .22, 'pixels sees the same edit.', size=12)
label(ax, 9.1, .63, 'Clone made before the edit', size=13, color=orange, weight='bold')
label(ax, 9.1, .22, 'Its values stay unchanged.', size=12)
finish(fig, 'tutorial_0_views_and_copies.png')

# 4. Image-label pairs stay paired during shuffling and batching.
fig, ax = canvas(12, 4.4)
label(ax, 2.2, 3.95, 'Dataset: sample pairs', size=15, weight='bold')
label(ax, 6.35, 3.95, 'DataLoader', size=15, weight='bold')
label(ax, 10.1, 3.95, 'A training batch', size=15, weight='bold')
for k in range(4):
    y0 = 2.98 - k*.72
    name, cl, color, fill = ('Star',0,blue,'#edf3f8') if k%2==0 else ('Heart',1,orange,'#fff1e6')
    box(ax, .35, y0, 3.6, .53, f'Image {k}: {name}     |     label {cl}', color=color, fill=fill, size=11)
box(ax, 5.02, 1.3, 2.65, 1.7, 'Shuffle pairs\n\nGroup 16 pairs', size=12)
arrow(ax, (4.12,2.17), (4.88,2.17))
arrow(ax, (7.83,2.17), (8.55,2.17))
box(ax, 8.7, 2.35, 3.0, .9, 'images\n[16, 1, 28, 28]', size=12)
box(ax, 8.7, 1.13, 3.0, .9, 'labels\n[16]', color=orange, fill='#fff7ef', size=12)
label(ax, 6.0, .23, 'The image and label at the same batch position still belong together.', size=13)
finish(fig, 'tutorial_0_dataloader.png')

# 5. Distinguish score calculation, gradient calculation, and parameter update.
fig, ax = canvas(12, 4.8)
label(ax, 6, 4.38, 'One training update', size=16, weight='bold')
box(ax, .3, 2.98, 2.65, .95, '1  zero_grad()\nClear old gradients', size=12)
box(ax, 4.6, 2.98, 2.8, .95, '2  model(images)\nCalculate scores', size=12)
box(ax, 8.9, 2.98, 2.8, .95, '3  loss_fn(logits, labels)\nMeasure error', size=11)
arrow(ax, (3.1,3.4), (4.44,3.4))
arrow(ax, (7.56,3.4), (8.74,3.4))
box(ax, 8.9, .98, 2.8, .95, '4  loss.backward()\nCalculate gradients', color=orange, fill='#fff7ef', size=12)
arrow(ax, (10.3,2.83), (10.3,2.08), color=orange)
box(ax, 4.6, .98, 2.8, .95, '5  optimizer.step()\nUpdate parameters', color=orange, fill='#fff7ef', size=12)
arrow(ax, (8.74,1.38), (7.56,1.38), color=orange)
box(ax, .3, .98, 2.65, .95, 'Ready for next batch', size=12)
arrow(ax, (4.44,1.38), (3.1,1.38))
arrow(ax, (1.62,2.08), (1.62,2.83))
label(ax, 6, .15, 'backward() computes gradients; only step() changes parameter values.', size=12)
finish(fig, 'tutorial_0_training_update.png')

# 6. Training and validation use the same model but different data and actions.
fig, ax = canvas(12, 4.6)
label(ax, 6, 4.15, 'One epoch: train, then validate', size=16, weight='bold')
box(ax, .25, 2.85, 2.6, .8, '128 training images\nwith labels', size=12)
box(ax, 4.1, 2.85, 3.45, .8, 'model.train()\n8 batches: predict + update', size=12)
box(ax, 9.0, 2.85, 2.7, .8, 'Updated model', size=12)
arrow(ax, (3.0,3.25), (3.94,3.25), color=blue)
arrow(ax, (7.71,3.25), (8.84,3.25), color=blue)
box(ax, .25, .95, 2.6, .8, '64 validation images\nwith labels', color=orange, fill='#fff7ef', size=12)
box(ax, 4.1, .95, 3.45, .8, 'Same model: eval()\nwith torch.no_grad()', color=orange, fill='#fff7ef', size=12)
box(ax, 9.0, .95, 2.7, .8, 'Validation accuracy\nNo parameter updates', color=orange, fill='#fff7ef', size=11)
arrow(ax, (3.0,1.35), (3.94,1.35), color=orange)
arrow(ax, (7.71,1.35), (8.84,1.35), color=orange)
arrow(ax, (10.35,2.73), (7.2,1.9), color=ink, connectionstyle='arc3,rad=.12')
label(ax, 6, .3, 'Training images change the model; validation images only check its predictions.', size=12)
finish(fig, 'tutorial_0_train_validation.png')

# 7. Stable colors identify each individual value through both arrangements.
colors = ['#dbeafe', '#c7d9f6', '#b4c8ef', '#ffedd5', '#fed7aa', '#fdba74']
fig, ax = canvas(12, 4.1)
label(ax, 1.7, 3.65, 'Original [2, 3]', size=15, weight='bold')
grid(ax, .65, 1.25, [[1,2,3],[4,5,6]], cell=.68, colors=[colors[:3],colors[3:]])
label(ax, 6.0, 3.65, 'reshape(3, 2)', size=15, weight='bold')
grid(ax, 5.3, .95, [[1,2],[3,4],[5,6]], cell=.68,
     colors=[colors[:2],colors[2:4],colors[4:]])
label(ax, 10.25, 3.65, 'permute(1, 0)', size=15, weight='bold')
grid(ax, 9.55, .95, [[1,4],[2,5],[3,6]], cell=.68,
     colors=[[colors[0],colors[3]],[colors[1],colors[4]],[colors[2],colors[5]]])
arrow(ax, (3.0,2.05),(4.9,2.05))
label(ax, 6.0, .35, 'Regroup entries\nin the same order', size=12)
label(ax, 10.25, .35, 'Swap row and\ncolumn axes', size=12)
label(ax, 1.7, .35, 'Both operations\nstart here', size=12)
ax.plot([2.8,3.4,8.85],[1.12,.8,.8],color=orange,linewidth=1.5)
arrow(ax, (8.85,.8),(9.4,1.15), color=orange)
finish(fig, 'tutorial_0_reshape_permute.png')

# 8. A new axis and a longer existing axis are different outcomes.
fig, ax = canvas(12, 3.8)
label(ax, 2.8, 3.38, 'stack([a, b])', size=15, weight='bold')
label(ax, 9.2, 3.38, 'cat([a, b])', size=15, weight='bold')
for left in [.55,6.55]:
    grid(ax, left, 2.0, [[1,2]], cell=.58, colors=[['#dbeafe']*2])
    grid(ax, left, .9, [[3,4]], cell=.58, colors=[['#ffedd5']*2])
    label(ax, left-.25,2.29,'a',size=14,color=blue)
    label(ax, left-.25,1.19,'b',size=14,color=orange)
    arrow(ax, (left+1.4,1.8),(left+2.55,1.8))
grid(ax, 3.5, 1.3, [[1,2],[3,4]], cell=.66, colors=[['#dbeafe']*2,['#ffedd5']*2])
grid(ax, 9.3, 1.7, [[1,2,3,4]], cell=.6, colors=[['#dbeafe']*2+['#ffedd5']*2])
label(ax, 4.15, .6, 'New axis\nshape [2, 2]', size=12)
label(ax, 10.5, .6, 'Extend existing axis\nshape [4]', size=12)
finish(fig, 'tutorial_0_stack_cat.png')

# 9. Show which entries are grouped by each reduction axis.
fig, ax = canvas(12, 3.9)
label(ax, 3.0,3.47,'mean(dim=1): across columns',size=14,weight='bold')
label(ax, 9.0,3.47,'mean(dim=0): down rows',size=14,weight='bold')
grid(ax, .65,1.3,[[1,2],[3,4]],cell=.75,colors=[['#dbeafe']*2,['#ffedd5']*2])
for yy, number, color in [(2.425,'1.5',blue),(1.675,'3.5',orange)]:
    arrow(ax,(2.35,yy),(3.55,yy),color=color)
    box(ax,3.75,yy-.31,1.1,.62,number,color=color,
        fill='#edf3f8' if color==blue else '#fff7ef',size=14)
label(ax, 2.85,.7,'One mean per row',size=13)
grid(ax, 8.25,1.7,[[1,2],[3,4]],cell=.75,
     colors=[['#dbeafe','#ffedd5'],['#dbeafe','#ffedd5']])
for xx,number,color in [(8.625,'2.0',blue),(9.375,'3.0',orange)]:
    arrow(ax,(xx,1.55),(xx,1.0),color=color)
    box(ax,xx-.33,.25,.66,.6,number,color=color,
        fill='#edf3f8' if color==blue else '#fff7ef',size=12)
label(ax, 6.0,.12,'Colors identify the entries averaged together.',size=11)
finish(fig, 'tutorial_0_mean_axes.png')
print('Created nine additional teaching diagrams.')
