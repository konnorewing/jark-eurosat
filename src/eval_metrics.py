"""F1 report and percent-only confusion matrix for the EuroSAT CNN.

Usage (after model.fit, on the labelled validation split):

    from eval_metrics import report_f1, plot_confusion_percent

    y_pred = model.predict(X_val, batch_size=256, verbose=0).argmax(axis=1)
    report_f1(y_val, y_pred, classes, csv_path='../results/f1_report.csv')
    plot_confusion_percent(y_val, y_pred, classes,
                           save_path='../results/confusion_matrix.png')
    plt.show()
"""
import csv

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, to_rgb
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# One-hue sequential ramp: light blue = low %, dark blue = high %
_RAMP = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b']
PCT_CMAP = LinearSegmentedColormap.from_list('pct_blues', _RAMP)

SURFACE = '#ffffff'      # background + gaps between cells
INK_DARK = '#0b0b0b'     # text on light cells
INK_LIGHT = '#f0efec'    # text on dark cells
INK_MUTED = '#52514e'    # axis labels / ticks


def _names(class_names):
    return [c.decode() if isinstance(c, bytes) else str(c) for c in class_names]


def report_f1(y_true, y_pred, class_names, csv_path=None):
    """Print per-class precision/recall/F1 plus macro & weighted F1; optionally save CSV."""
    names = _names(class_names)
    labels = np.arange(len(names))

    print(classification_report(y_true, y_pred, labels=labels, target_names=names,
                                digits=3, zero_division=0))

    macro = f1_score(y_true, y_pred, labels=labels, average='macro', zero_division=0)
    weighted = f1_score(y_true, y_pred, labels=labels, average='weighted', zero_division=0)
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)

    print(f'Macro F1:    {macro:.4f}')
    print(f'Weighted F1: {weighted:.4f}')
    worst = np.argsort(per_class)[:3]
    print('Lowest F1:   ' + ', '.join(f'{names[i]} ({per_class[i]:.3f})' for i in worst))

    if csv_path:
        rep = classification_report(y_true, y_pred, labels=labels, target_names=names,
                                    output_dict=True, zero_division=0)
        with open(csv_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['class', 'precision', 'recall', 'f1', 'support'])
            for name in names + ['macro avg', 'weighted avg']:
                r = rep[name]
                w.writerow([name, f"{r['precision']:.4f}", f"{r['recall']:.4f}",
                            f"{r['f1-score']:.4f}", int(r['support'])])
        print(f'wrote {csv_path}')

    return {'macro_f1': macro, 'weighted_f1': weighted,
            'per_class_f1': dict(zip(names, per_class))}


def _luminance(rgb):
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def _contrast(l1, l2):
    hi, lo = max(l1, l2), min(l1, l2)
    return (hi + 0.05) / (lo + 0.05)


def plot_confusion_percent(y_true, y_pred, class_names, save_path=None, gamma=1.0):
    """Row-normalised confusion matrix: each cell = % of that TRUE class predicted as
    the column class, so every row sums to 100 and the diagonal is per-class recall.
    Cells are coloured by the same percentage; exact-zero cells are left blank.

    gamma=1.0 keeps the colour scale linear. A value like 0.4 stretches the low end
    so 1-5 % confusions show up as visibly darker than 0.2 % ones (the colorbar
    ticks still read in true percent)."""
    names = _names(class_names)
    n = len(names)
    labels = np.arange(n)
    cm = confusion_matrix(y_true, y_pred, labels=labels, normalize='true') * 100

    side = 0.8 * n + 2.5
    fig, ax = plt.subplots(figsize=(side + 1.2, side), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    norm = Normalize(0, 100) if gamma == 1.0 else PowerNorm(gamma, vmin=0, vmax=100)
    mesh = ax.pcolormesh(cm, cmap=PCT_CMAP, norm=norm,
                         edgecolors=SURFACE, linewidth=2)
    ax.invert_yaxis()                     # first class at the top
    ax.set_aspect('equal')

    lum_dark, lum_light = _luminance(to_rgb(INK_DARK)), _luminance(to_rgb(INK_LIGHT))
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v == 0:
                continue
            text = '<0.1%' if v < 0.05 else f'{v:.1f}%'
            bg = _luminance(PCT_CMAP(norm(v))[:3])
            ink = INK_DARK if _contrast(bg, lum_dark) >= _contrast(bg, lum_light) else INK_LIGHT
            ax.text(j + 0.5, i + 0.5, text, ha='center', va='center',
                    fontsize=9, color=ink,
                    fontweight='bold' if i == j else 'normal')

    ax.set_xticks(labels + 0.5, names, rotation=40, ha='right', rotation_mode='anchor')
    ax.set_yticks(labels + 0.5, names)
    ax.tick_params(length=0, colors=INK_MUTED, labelsize=9)
    ax.set_xlabel('Predicted class', color=INK_MUTED, labelpad=8)
    ax.set_ylabel('True class', color=INK_MUTED, labelpad=8)
    ax.set_title('Confusion matrix (% of each true class)', loc='left',
                 color=INK_DARK, fontsize=12, pad=12)
    for s in ax.spines.values():
        s.set_visible(False)

    cbar = fig.colorbar(mesh, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label('% of true class', color=INK_MUTED)
    cbar.ax.tick_params(length=0, colors=INK_MUTED, labelsize=8)
    cbar.outline.set_visible(False)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight', facecolor=SURFACE)
        print(f'wrote {save_path}')
    return fig
