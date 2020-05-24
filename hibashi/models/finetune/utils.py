import itertools
import re
from textwrap import wrap

import matplotlib.pyplot as plt
import numpy as np


def plot_confusion_matrix_validation(cm: np.ndarray, labels: list, figsize=(4, 4)):
    """
    Generates the figures for aggregated results for validation.
    :param cm: aggregates confusion matrix values
    :param labels: of class labels
    :param figsize: size of matplotlib figure, needs to be large enough to accommodate all categories
    :return: (matplotlibt object) figure
    """
    np.set_printoptions(precision=2)

    fig = plt.figure(figsize=figsize, dpi=320, facecolor='w', edgecolor='k')
    ax = fig.add_subplot(1, 1, 1)
    im = ax.imshow(cm, cmap='Oranges')

    classes = [re.sub(r'([a-z](?=[A-Z])|[A-Z](?=[A-Z][a-z]))', r'\1 ', x) for x in labels]
    classes = ['\n'.join(wrap(l, 40)) for l in classes]

    tick_marks = np.arange(len(classes))

    ax.set_xlabel('Predicted [count]', fontsize=5)
    ax.set_xticks(tick_marks)
    c = ax.set_xticklabels(classes, fontsize=4, rotation=-90, ha='center')
    ax.xaxis.set_label_position('bottom')
    ax.xaxis.tick_bottom()

    ax.set_ylabel('True Label [count]', fontsize=5)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes, fontsize=4, va='center')
    ax.yaxis.set_label_position('left')
    ax.yaxis.tick_left()

    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], 'd') if cm[i, j] != 0 else '.', horizontalalignment="center", fontsize=4,
                verticalalignment='center', color="black")
    fig.set_tight_layout(True)

    return fig