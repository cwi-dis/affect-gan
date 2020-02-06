import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataloader import Dataloader
from matplotlib.widgets import MultiCursor

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



if __name__ == '__main__':
    dataloader = Dataloader(features=["ecg", "bvp", "gsr", "rsp", "skt"], label=["subject", "video"], normalized=False)
    data = dataloader("inspect", 1)

    plot_signals(data)

    #plot_label_heatmap(labels)
    #labels = collect_labels(data)
