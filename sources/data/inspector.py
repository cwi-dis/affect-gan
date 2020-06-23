import numpy as np
import os
import io
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
import tensorflow_datasets as tfds
import itertools
from sklearn.manifold import TSNE

from data.dataloader import Dataloader
from data.datagenerator import DatasetGenerator
from models.Blocks import get_positional_encoding
from matplotlib.widgets import MultiCursor, Slider, Button, TextBox
from util.misc import init_tf_gpus

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

def plot_generated_signals(signals_0, signals_1):
    n_samples=5#len(signals_0)
    colors = ["blue", "red", "green"]
    x = range(len(signals_0[0]))
    n_signals = len(signals_0[0][0])
    fig, axs = plt.subplots(n_samples, 2, sharex=True)
    for sample in range(n_samples):
        for sig in range(n_signals):
            axs[sample, 0].plot(x, signals_0[sample, :, sig], color=colors[sig])
            axs[sample, 1].plot(x, signals_1[sample, :, sig], color=colors[sig])
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

    filepath = "../../plots/data/%s.pkl" % filename

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

    labels.to_pickle("../../plots/data/%s.pkl" % filename)

    return collect_extended_labels(data, filename)


def plot_label_heatmap(labels):
    #ax1 = sns.jointplot(data=labels, x="valence", y="arousal", kind="hex", color="k")
    cmap = sns.cubehelix_palette(as_cmap=True, dark=0, light=1, reverse=True)
    sns.kdeplot(labels.valence, labels.arousal, cmap=cmap, n_levels=60, shade=True)
    plt.show()


def plot_signal(signal):
    colors = ["blue", "red", "green"]
    fig = plt.figure(figsize=(8, 2))
    x = range(len(signal))
    n_signals = signal.shape[-1]
    axs = fig.add_subplot(111)
    #for i in range(n_signals):
    #    axs.plot(x, signal[:, i], color=colors[i])
    for i in range(n_signals):
        f, = axs.plot(x, signal[:, i], color=colors[i])
    plt.show()

def plot_signals(data_source, generated=False, disc=None, subject_seed=None):
    if subject_seed is None:
        if generated:
            data = data_source()
        else:
            data = data_source
        for signal, label, subject in data.take(100):
            print("###########################")
            if disc:
                _, v, labp, subp = disc(signal)
                print(v)
                print(np.around(labp, 2))
                #print(np.around(subp, 2))
            print(label)
            print(np.around(subject, 2))
            plot_signal(signal[0])
    else:
        while(True):
            signal, label = data_source.get(arousal_value=None, subject_value=subject_seed, sub0=None, sub1=None)
            if disc:
                _, v, labp, subp = disc(signal)
                print(v)
                print(np.around(labp, 2))
                print(np.around(subp, 2))
            print(label)
            plot_signal(signal[0])


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

    g = sns.FacetGrid(extended_labels, col="subject",  hue="video", palette=sns.color_palette(color_pallete, n_colors=11), col_wrap=2, legend_out=True)
    g.map(sns.lineplot, "valence", "arousal", sort=False)
    g.add_legend(title="VideoID")
    plt.show()

def video_viz(extended_labels):
    g = sns.FacetGrid(extended_labels, col="video", col_wrap=4)
    g.map(sns.scatterplot, "valence", "arousal")
    g.add_legend(title="SubjectID")
    plt.show()

def positional_ecoding_viz():
    pos_encoding = get_positional_encoding(250, 40)
    print(pos_encoding.shape)

    plt.pcolormesh(pos_encoding[0], cmap='RdBu')
    plt.xlabel('Depth')
    plt.xlim((0, 40))
    plt.ylabel('Position')
    plt.colorbar()
    plt.show()

def tsna_visualization(data):
    data_sample = data.take(10000)

    first = True
    colors = []
    for signal, label in data_sample:
        if first:
            tsne_data = tf.reshape(signal, shape=(1,-1))
            first = False
        else:
            tsne_data = tf.concat([tsne_data, tf.reshape(signal, shape=(1, -1))], axis=0)
        colors.append("blue" if label.numpy() == 0 else "red")

    tsne = TSNE(n_components=2, perplexity=50, verbose=1, n_iter=5000)
    tsne_results = tsne.fit_transform(tsne_data)

    f, ax = plt.subplots(1)
    plt.scatter(tsne_results[:,0], tsne_results[:,1], c=colors, alpha=0.2)
    ax.legend()
    plt.show()

def interactive_signal_plot(datagen, disc):
    mpl.rcParams['font.family'] = 'Avenir'
    mpl.rcParams['font.size'] = 8
    mpl.rcParams['axes.linewidth'] = 2
    mpl.rcParams['axes.spines.top'] = False
    mpl.rcParams['axes.spines.right'] = False
    mpl.rcParams['xtick.major.size'] = 10
    mpl.rcParams['xtick.major.width'] = 2
    mpl.rcParams['ytick.major.size'] = 10
    mpl.rcParams['ytick.major.width'] = 2
    colors = ["blue", "red", "green"]
    fig = plt.figure(figsize=(8, 4))

    ax = fig.add_subplot(111)
    fig.subplots_adjust(bottom=0.2, top=0.75)

    ax_subject = fig.add_axes([0.3, 0.85, 0.4, 0.05])
    ax_subject.spines['top'].set_visible(True)
    ax_subject.spines['right'].set_visible(True)

    ax_arousal = fig.add_axes([0.3, 0.92, 0.4, 0.05])
    ax_arousal.spines['top'].set_visible(True)
    ax_arousal.spines['right'].set_visible(True)

    ax_seed_button = fig.add_axes([0.75, 0.85, 0.1, 0.08])
    ax_seed_button.spines['top'].set_visible(True)
    ax_seed_button.spines['right'].set_visible(True)

    ax_text0 = fig.add_axes([0.1, 0.92, 0.1, 0.05])
    ax_text0.spines['top'].set_visible(True)
    ax_text0.spines['right'].set_visible(True)

    ax_text1 = fig.add_axes([0.1, 0.85, 0.1, 0.05])
    ax_text1.spines['top'].set_visible(True)
    ax_text1.spines['right'].set_visible(True)

    slider_subject = Slider(ax=ax_subject, label='Subject ', valmin=0, valmax=1.0, valinit=0,
                 valfmt='%1.2f', valstep=0.05, facecolor='#cc7000')
    slider_arousal = Slider(ax=ax_arousal, label='Arousal ', valmin=0, valmax=1.0, valinit=0,
                 valfmt='%1.2f', valstep=0.05, facecolor='#cc7000')
    seed_button = Button(ax=ax_seed_button, label="New Seed")
    sub_0_text = TextBox(ax=ax_text0, label="Sub. 0 ID", initial="4", label_pad=0.05)
    sub_1_text = TextBox(ax=ax_text1, label="Sub. 1 ID", initial="18", label_pad=0.05)

    sig = datagen.get(arousal_value=0, subject_value=0, sub0=4, sub1=18)
    _, _, class_pred2, subj_pred = disc(sig)

    scale = range(500)
    n_signals = len(sig[0, 0])

    fig_sig = []
    for i in range(n_signals):
        f, = ax.plot(scale, sig[0, :, i], color=colors[i])
        fig_sig.append(f)

    def slider_update(val):
        arousal_val = slider_arousal.val
        subject_val = slider_subject.val
        sub0 = int(sub_0_text.text)
        sub1 = int(sub_1_text.text)
        new_sig = datagen.get(arousal_value=arousal_val, subject_value=subject_val, sub0=sub0, sub1=sub1, noise_seed_reuse=True)
        _, _, class_pred2, subj_pred = disc(new_sig)
        for i in range(n_signals):
            fig_sig[i].set_data(scale, new_sig[0, :, i])
        fig.canvas.draw_idle()

    def seed_button_click(v):
        arousal_val = slider_arousal.val
        subject_val = slider_subject.val
        sub0 = int(sub_0_text.text)
        sub1 = int(sub_1_text.text)
        new_sig = datagen.get(arousal_value=arousal_val, subject_value=subject_val, sub0=sub0, sub1=sub1, noise_seed_reuse=False)
        _, _, class_pred2, subj_pred = disc(new_sig)
        for i in range(n_signals):
            fig_sig[i].set_data(scale, new_sig[0, :, i])
        fig.canvas.draw_idle()


    slider_subject.on_changed(slider_update)
    slider_arousal.on_changed(slider_update)
    seed_button.on_clicked(seed_button_click)
    sub_0_text.on_submit(slider_update)
    sub_1_text.on_submit(slider_update)

    plt.show()

def visualize_tsna_denselayer(generator, discriminator, real_data):
    real_activations = []
    real_subjects = []
    lcolors = []
    pmarkers = []
    cm = plt.cm.get_cmap('seismic')
    for data, label, subject in real_data:
        dense_act, _, pl ,_ = discriminator(data)
        real_activations.append(dense_act)
        #lcolors.append("blue" if label.numpy()[0,0] == 0 else "red")
        lcolors.append(label.numpy()[0,0])
        real_subjects.append(subject)


    real_activations = tf.reshape(real_activations, (-1, 256))

    tsne = TSNE(n_components=2, perplexity=25, verbose=1, n_iter=5000)
    tsne_results = tsne.fit_transform(real_activations)

    f, ax = plt.subplots(1)
    sc = plt.scatter(tsne_results[:, 0], tsne_results[:, 1], c=lcolors, vmin=0, vmax=10, cmap=cm, alpha=0.4)
    plt.colorbar(sc)
    ax.legend()
    plt.show()

def map_subject_to_pdf(data, discriminator):
    pdf = None
    count = 0
    for d, l, s in data:
        _, _, _, ps = discriminator(d)
        if pdf is None:
            pdf = ps
        else:
            pdf = tf.concat([pdf, ps], axis=0)
        count += 1

    pdf_mean = tf.reduce_mean(pdf, axis=0)
    pdf_std = tf.math.reduce_std(pdf, axis=0)

    return pdf


if __name__ == '__main__':
    os.chdir("./..")
    init_tf_gpus()
    dataloader = Dataloader("5000d", features=["ecg", "gsr"],
                            label=["arousal"],
                            normalized=True, continuous_labels=True)
    data = dataloader("inspect", 1, leave_out=18)
    #datagenerator = DatasetGenerator(batch_size=1,
    #                                 path="../Logs/loso-wgan-class-subject/subject-4-out",
    #                                 subject_conditioned=True,
    #                                 class_categorical_sampling=False,
    #                                 subject_categorical_sampling=False,
    #                                 discriminator_class_conditioned=False,
    #                                 no_subject_output=False,
    #                                 argmaxed_label=True
    #                                 )
    #datag = datagenerator()
    #disc = tf.keras.models.load_model("../Logs/loso-wgan-class-subject/subject-4-out/model_dis")


    datagenerator = DatasetGenerator(batch_size=1,
                            path="../Logs/loso-wgan-class-subject/subject-18-out",
                            subject_conditioned=True,
                            categorical_sampling=False,
                            no_subject_output=False,
                            argmaxed_label=True
                           )
    disc = tf.keras.models.load_model("../Logs/loso-wgan-class-subject/subject-18-out/model_dis")

    #visualize_tsna_denselayer(gen, disc, data)

    #labels = collect_labels(data)
    #extended_labels = collect_extended_labels(data, "extended_labels_CASE", force_recollect=True)
    #print(extended_labels.describe())

    #valence_arousal_viz(extended_labels)
    #video_subject_viz(extended_labels)
    #video_viz(extended_labels)

    subject_seed = map_subject_to_pdf(data, disc)
    plot_signals(datagenerator, generated=True, disc=disc, subject_seed=subject_seed)
    #interactive_signal_plot(datagenerator, disc)
    #positional_ecoding_viz()


    #tsna_visualization(data)

    #plot_label_heatmap(labels)
