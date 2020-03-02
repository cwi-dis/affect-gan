import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataloader import Dataloader
from matplotlib.widgets import MultiCursor

color_pallete = ["#e6194B", "#ffe119", "#4363d8", "#f58231", "#42d4f4", "#f032e6", "#fabebe", "#469990", "#e6beff", "#9A6324", "#000000", "#800000", "#aaffc3", "#000075", "#a9a9a9", "#ffffff", "#3cb44b"]

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

def plot_signals(data):
    for signals, label in data.take(10):
        print(label)
        x = range(len(signals))
        fig, (ax0, ax1, ax2, ax3, ax4) = plt.subplots(5, sharex=True)
        ax0.plot(x, signals[:, 0])
        ax1.plot(x, signals[:, 1])
        ax2.plot(x, signals[:, 2])
        ax3.plot(x, signals[:, 3])
        ax4.plot(x, signals[:, 4])
        multi = MultiCursor(fig.canvas, (ax0, ax1, ax2, ax3, ax4), color='r', lw=1)
        plt.show()

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
