"""Illustrative workflow, not experimental data; text is 18pt at final width."""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


def draw_concept(save):
    """Show reference selection, short vectors and a size-controlled assignment."""
    blue, orange, green = '#0072B2', '#D55E00', '#009E73'
    fig = plt.figure(figsize=(6.5, 3.6))
    centers = [.16, .50, .84]
    titles = ['Select a few\nsamples', 'Build short\nvectors', 'Form the\nsets']
    for x, title in zip(centers, titles):
        fig.text(x, .94, title, ha='center', va='top', fontsize=18)

    left = fig.add_axes([.025, .35, .27, .40])
    xy = np.array([(x, y) for y in [3, 2, 1, 0] for x in [0, 1, 2]])
    left.scatter(xy[:, 0], xy[:, 1], s=65, color='#8498AB', zorder=2)
    selected = [0, 4, 8]
    left.scatter(xy[selected, 0], xy[selected, 1], s=230, facecolors='none',
                 edgecolors=orange, linewidths=2.2, zorder=3)
    left.set(xlim=(-.5, 2.5), ylim=(-.6, 3.6));left.set_axis_off()

    # Three columns correspond to three selected reference samples. The drawing
    # illustrates factor storage; no fitted model or benchmark is represented.
    rng = np.random.default_rng(4)
    B = rng.uniform(.15, 1, (12, 3))
    middle = fig.add_axes([.405, .35, .19, .40])
    middle.imshow(B, cmap='Blues', vmin=0, vmax=1, aspect='auto', interpolation='nearest')
    middle.set_xticks(np.arange(-.5, 3, 1), minor=True)
    middle.set_yticks(np.arange(-.5, 12, 1), minor=True)
    middle.grid(which='minor', color='white', linewidth=1.2)
    middle.tick_params(which='both', left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in middle.spines.values():spine.set_visible(False)

    right = fig.add_axes([.68, .35, .30, .40]);right.set(xlim=(0, 1), ylim=(0, 1));right.set_axis_off()
    for y, label, count, color in [(.81, 'Train', 8, blue), (.49, 'Validate', 2, orange), (.17, 'Test', 2, green)]:
        right.add_patch(FancyBboxPatch((.015, y-.13), .97, .26,
            boxstyle='round,pad=0.01,rounding_size=0.025', facecolor='#F5F8FA', edgecolor='#C7D1DA', lw=1))
        right.text(.06, y, label, fontsize=18, va='center')
        points = np.array([(.72+(j%2)*.14, y+.075-(j//2)*.05) for j in range(count)])
        right.scatter(points[:, 0], points[:, 1], s=22, color=color, zorder=3)

    for start, end in [(.305, .385), (.615, .665)]:
        fig.add_artist(FancyArrowPatch((start, .55), (end, .55), transform=fig.transFigure,
            arrowstyle='-|>', mutation_scale=16, lw=1.8, color='#425466'))

    fig.text(.5, .235, 'Sum the vectors within each set', ha='center', va='center', fontsize=18)
    fig.text(.5, .13, 'Estimate similarity between sets', ha='center', va='center', fontsize=18)
    save(fig, 'concept')
