import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.dataloader import Dataloader


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

if __name__ == '__main__':
    dataloader = Dataloader(label=["arousal", "valence"])
    data = dataloader(1, "inspect")

    labels = collect_labels(data)
    print(labels)

    plot_label_heatmap(labels)