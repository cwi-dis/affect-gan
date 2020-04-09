import numpy as np
import os
import io
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import itertools

from data.dataloader import Dataloader
from matplotlib.widgets import MultiCursor

color_pallete = ["#e6194B", "#ffe119", "#4363d8", "#f58231", "#42d4f4", "#f032e6", "#fabebe", "#469990", "#e6beff", "#9A6324", "#000000", "#800000", "#aaffc3", "#000075", "#a9a9a9", "#ffffff", "#3cb44b"]


def plot_to_image(figure):
    """Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call."""
    # Save the plot to a PNG in memory.
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    # Convert PNG buffer to TF image
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    return image


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
    cm (array, shape = [n, n]): a confusion matrix of integer classes
    class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(2, 2))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_generated_signals(signals, n_signals):
    sig1, sig2 = signals[0], signals[1]
    x = range(len(sig1))
    fig, axs = plt.subplots(n_signals, 2)
    for sig in range(2):
        if n_signals == 1:
            axs[sig].plot(x, signals[sig, :, 0])
        else:
            for phys_sig in range(n_signals):
                axs[phys_sig, sig].plot(x, signals[sig, :, phys_sig])
    plt.tight_layout()
    return fig




def collect_labels(data):
    arousal = []
    valence = []
    for _, label in data:
        arousal.append(label.numpy()[0])
        valence.append(label.numpy()[1])

    labels = {"arousal": arousal,
              "valence": valence}

    labels = pd.DataFrame(labels)

    return labels

def collect_extended_labels(data, filename, force_recollect=False):

    filepath = f"../../plots/data/{filename}.pkl"

    if os.path.exists(filepath) and not force_recollect:
        labels = pd.read_pickle(filepath)
        labels["subject"] = labels.subject.astype('int32')
        labels["video"] = labels.video.astype('int32')
        return labels

    arousal = []
    valence = []
    subject = []
    video = []
    for _, label in data:
        arousal.append(label.numpy()[0])
        valence.append(label.numpy()[1])
        subject.append(label.numpy()[2])
        video.append(label.numpy()[3])

    labels = {"arousal": arousal,
              "valence": valence,
              "subject": subject,
              "video": video
              }

    labels = pd.DataFrame(labels)

    labels.to_pickle(f"../../plots/data/{filename}.pkl")

    return collect_extended_labels(data, filename)


def plot_label_heatmap(labels):
    #ax1 = sns.jointplot(data=labels, x="valence", y="arousal", kind="hex", color="k")
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    sns.kdeplot(labels.valence, labels.arousal, cmap=cmap, n_levels=60, shade=True)
    plt.show()


def plot_signal(signal):
    x = range(len(signal))
    fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, sharex=True)
    ax0.plot(x, signal[:, 0])
    ax1.plot(x, signal[:, 1])
    ax2.plot(x, signal[:, 2])
    ax3.plot(x, signal[:, 3])
    ax4.plot(x, signal[:, 4])
    multi = MultiCursor(fig.canvas, (ax0, ax1, ax2, ax3, ax4), color='r', lw=1)
    plt.show()

def plot_signals(data):
    for signal, label in data.take(10):
        print(label)
        plot_signal(signal)


def valence_arousal_viz(extended_labels):
    current_palette = sns.color_palette("bright", n_colors=11)
    ax0 = sns.scatterplot(data=extended_labels, x="valence", y="arousal", hue="video", legend="full", palette=current_palette)
    plt.show()
    ax0 = sns.scatterplot(data=extended_labels, x="valence", y="arousal", hue="subject", legend="full", palette=sns.color_palette(color_pallete, n_colors=30))
    plt.show()

def video_subject_viz(extended_labels):
    #fig, axes = plt.subplots(5,6, sharex="col", sharey="row")
    #
    #for subject, data in extended_labels.groupby('subject', as_index=False):
    #    for video, video_data in data.groupby("video", as_index=False):
    #        axes[(subject-1) // 5, (subject-1) % 5].lineplot(data=video_data, x="valence", y="arousal", sort="false", hue="video", palette=sns.color_palette("bright, n_colors=11"))

    g = sns.FacetGrid(extended_labels, col="subject",  hue="video", palette=sns.color_palette(color_pallete, n_colors=11), col_wrap=5, legend_out=True)
    g.map(sns.lineplot, "valence", "arousal", sort=False)
    g.add_legend(title="VideoID")
    plt.show()

def video_viz(extended_labels):
    g = sns.FacetGrid(extended_labels, col="video", col_wrap=4)
    g.map(sns.scatterplot, "valence", "arousal")
    g.add_legend(title="SubjectID")
    plt.show()

if __name__ == '__main__':
    os.chdir("./..")
    dataloader = Dataloader("5000d", features=["bvp", "ecg", "rsp", "gsr", "skt"], label=["arousal", "valence"], normalized=True)
    data = dataloader("inspect", 1)

    #labels = collect_labels(data)
    #extended_labels = collect_extended_labels(data, "extended_labels_CASE")
    #print(extended_labels.describe())

    #valence_arousal_viz(extended_labels)
    #video_subject_viz(extended_labels)
    #video_viz(extended_labels)

    plot_signals(data)

    #plot_label_heatmap(labels)
