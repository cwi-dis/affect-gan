import os
import glob
import tensorflow as tf
import pandas as pd
from scipy.signal import decimate
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np
import pickle

from util.misc import di, dl
from tensorflow.python.ops import math_ops

def window_dataset(dataset):
    window_ds = dataset.window(1000, shift=1000)

    def sub_to_batch(sub):
        return sub.batch(1000, drop_remainder=True)

    windows = window_ds.flat_map(sub_to_batch)
    return windows

@tf.function
def with_categoric_labels(features, label, subject, threshold=5.0):
    return features, math_ops.cast(label > threshold, label.dtype), subject

class Dataloader(object):

    def __init__(self, datasetID, features=None, label=None, continuous_labels=True, normalized=True):
        if features is None:
            features = ["bvp"]
        self.feature_index = {
            "bvp": 0,
            "ecg": 1,
            "gsr": 2,
            "rsp": 3,
            "skt": 4
        }
        path = "../Dataset/CASE_dataset/tfrecord_%s/" % datasetID
        if path is None or not os.path.exists(path):
            raise Exception("Data path does not exist:" + os.curdir +":"+path)

        self.path = path
        self.next_element = None
        #self.window_size = window_size
        self.features = features
        if label is None:
            self.labels = ["arousal"]
        else:
            self.labels = label
        self.continuous_labels = continuous_labels
        self.normalized = normalized
        #self.range_clipped = range_clipped  #range clipping now included in normalization

        #self.means = pd.read_csv("../Dataset/CASE_dataset/stats/mean.csv", header=None, index_col=0, squeeze=True).astype("float32")
        #self.vars = pd.read_csv("../Dataset/CASE_dataset/stats/var.csv", header=None, index_col=0, squeeze=True).astype("float32")
        #self.minmax = pd.read_csv("../Dataset/CASE_dataset/stats/minmax.csv", header=0, index_col=0, squeeze=True).astype("float32")

        with open("../Dataset/CASE_dataset/stats/subj_mean.pickle", "rb") as handle:
            self.subj_means = pd.DataFrame.from_dict(pickle.load(handle)).astype("float32")
        with open("../Dataset/CASE_dataset/stats/subj_minmax.pickle", "rb") as handle:
            self.subj_minmax = pd.DataFrame.from_dict(pickle.load(handle)).astype("float32")

        self.excluded_label = tf.constant(5, tf.float32)
        self.subject_labels = list(range(1, 31))

    def _decode(self, serialized_example):
        decoded_example = tf.io.parse_single_example(serialized_example,
                                                     features={
                                                         'Subject': tf.io.FixedLenFeature([], tf.int64),
                                                         'VideoID': tf.io.FixedLenFeature([], tf.int64),
                                                         'ecg': tf.io.FixedLenSequenceFeature([], tf.float32,
                                                                                              allow_missing=True),
                                                         'bvp': tf.io.FixedLenSequenceFeature([], tf.float32,
                                                                                              allow_missing=True),
                                                         'gsr': tf.io.FixedLenSequenceFeature([], tf.float32,
                                                                                              allow_missing=True),
                                                         'rsp': tf.io.FixedLenSequenceFeature([], tf.float32,
                                                                                              allow_missing=True),
                                                         'skt': tf.io.FixedLenSequenceFeature([], tf.float32,
                                                                                              allow_missing=True),
                                                         'valence': tf.io.FixedLenSequenceFeature([], tf.float32,
                                                                                                  allow_missing=True),
                                                         'arousal': tf.io.FixedLenSequenceFeature([], tf.float32,
                                                                                                  allow_missing=True)
                                                     })

        video = tf.cast(decoded_example["VideoID"], tf.float32)

        subject = tf.cast(decoded_example["Subject"], tf.float32)
        subject_i = tf.cast(subject, tf.int32)

        features = []
        if self.normalized:
            #Old Normalization:
            #for feature in self.features:
            #    znorm = (decoded_example[feature] - self.means[feature]) / tf.sqrt(self.vars[feature])
            #    if self.range_clipped:
            #        znorm = 2 * (znorm - self.minmax[min][feature]) / (self.minmax[max][feature] - self.minmax[min][feature]) - 1
            #    features.append(znorm)

            #Per subject Normalization:
            subj_mean = tf.gather(self.subj_means, indices=subject_i-1, axis=-1)
            subj_minmax = tf.gather(self.subj_minmax, indices=subject_i-1, axis=-1)
            for feature in self.features:
                i = self.feature_index[feature]
                standardized = decoded_example[feature] - subj_mean[i]
                standardized = -1 + 2 * (standardized + subj_minmax[i]) / (subj_minmax[i] + subj_minmax[i])
                features.append(standardized)
        else:
            for feature in self.features:
                features.append(decoded_example[feature])

        labels = []
        for label in self.labels:
            if label is "subject":
                labels.append(subject)
            elif label is "video":
                labels.append(video)
            else:
                value = tf.reduce_mean(decoded_example[label][-50:])
                labels.append(value)

        features = tf.stack(features, axis=1)
        features.set_shape([500, len(self.features)])

        return features, labels, video, subject

    def __call__(self, mode, batch_size=64, leave_out=None, one_hot=False, repeat=False):

        modes = ["train", "test", "eval", "inspect", "test_eval", "gan", "cgan"]
        if mode not in modes:
            raise Exception("mode not found! supported modes are %s" % modes)
        #if mode is "eval" and leave_out is None:
        #    raise Exception("leave-one-out evaluation undefined!")

        if leave_out is None:
            train_subject_ids = self.subject_labels[:28]
            eval_subject_ids = self.subject_labels[28:]
        else:
            train_subject_ids = self.subject_labels[:leave_out-1] + self.subject_labels[leave_out:]
            eval_subject_ids = self.subject_labels[leave_out-1:leave_out]

        if (mode is "train") or (mode is "gan"):
            files = [glob.glob("%ssub_%d.tfrecord" % (self.path, num)) for num in train_subject_ids]
        elif (mode is "test") or (mode is "test_eval") or (mode is "inspect" and leave_out is not None):
            files = [glob.glob("%ssub_%d.tfrecord" % (self.path, num)) for num in eval_subject_ids]
        else:
            files = [glob.glob("%s*.tfrecord" % self.path)]

        #print("Files loaded in mode %s:"%files)
        files = tf.data.Dataset.from_tensor_slices(files)

        dataset = files.interleave(lambda f: tf.data.TFRecordDataset(f).map(self._decode, num_parallel_calls=1),
                                   num_parallel_calls=tf.data.experimental.AUTOTUNE, block_length=16)
        #dataset = dataset.map(self._decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.filter(lambda _, __, video, ___: tf.less(video, 10))
        dataset = dataset.map(lambda features, labels, video, subject: (features, labels, subject))
        if mode is not "inspect":
            dataset = dataset.filter(lambda _, label, __: tf.reduce_all(tf.greater(tf.abs(label - self.excluded_label), 0.05)))
        if self.normalized:
            dataset = dataset.filter(lambda features, _, __: tf.less_equal(tf.reduce_max(features), 1.2) and tf.less_equal(tf.abs(tf.reduce_min(features)), 1.2))

        if not self.continuous_labels:
            dataset = dataset.map(with_categoric_labels)

        if one_hot and (mode is not "gan"):
            dataset = dataset.map(lambda data, label, subject: (data, tf.squeeze(tf.one_hot(tf.cast(label, tf.int32), depth=2))))

        if mode == "train":
            trainset, evalset = split_dataset(dataset, validation_data_fraction=0.1)
            if repeat:
                trainset = trainset.repeat()
            trainset = trainset.shuffle(buffer_size=30000)
            trainset = trainset.batch(batch_size)
            trainset = trainset.prefetch(1)

            evalset = evalset.batch(batch_size)

            return trainset, evalset

        if mode is "gan":
            dataset = dataset.map(lambda features, labels, subject: (features, labels, tf.cond(tf.greater(subject, leave_out), lambda: subject - 2, lambda: subject - 1)))
            dataset = dataset.repeat().shuffle(buffer_size=30000)

        if mode is "inspect":
            dataset = dataset.shuffle(buffer_size=10000)
            return dataset

        if mode is "test":
            dataset = dataset.batch(batch_size)
            return dataset


        #Fix for dual output classification
        #if len(self.labels) == 2:
        #    dataset = dataset.map(lambda data, labels: (data, (labels[0], labels[1])))



def split_dataset(dataset: tf.data.Dataset, validation_data_fraction: float):
    """
    Splits a dataset of type tf.data.Dataset into a training and validation dataset using given ratio. Fractions are
    rounded up to two decimal places.
    @param dataset: the input dataset to split.
    @param validation_data_fraction: the fraction of the validation data as a float between 0 and 1.
    @return: a tuple of two tf.data.Datasets as (training, validation)
    """

    validation_data_percent = round(validation_data_fraction * 100)
    if not (0 <= validation_data_percent <= 100):
        raise ValueError("validation data fraction must be ∈ [0,1]")

    dataset = dataset.enumerate()
    train_dataset = dataset.filter(lambda f, data: f % 100 > validation_data_percent)
    validation_dataset = dataset.filter(lambda f, data: f % 100 <= validation_data_percent)

    # remove enumeration
    train_dataset = train_dataset.map(lambda f, data: data)
    validation_dataset = validation_dataset.map(lambda f, data: data)

    return train_dataset, validation_dataset

if __name__ == '__main__':
    os.chdir("./..")
    labels = ["ecg", "gsr"]
    d = Dataloader("5000d", labels, label=["subject"],
                   normalized=True, continuous_labels=True)
    d = d("inspect", 1, 6)

    i = 0
    for _ in d:
        i += 1
    print(i)
#with open("../Dataset/CASE_dataset/stats/subj_mean.pickle", "rb") as handle:
    #    subj_means = pd.DataFrame.from_dict(pickle.load(handle)).astype("float32")
    #with open("../Dataset/CASE_dataset/stats/subj_minmax.pickle", "rb") as handle:
    #    subj_minmax = pd.DataFrame.from_dict(pickle.load(handle)).astype("float32")
#
    #print("Yes")


    #with open("../Dataset/CASE_dataset/stats/mean.pickle", "rb") as f:
    #    dmean = pickle.load(f)
    #with open("../Dataset/CASE_dataset/stats/max.pickle", "rb") as f:
    #    dmax = pickle.load(f)
    #with open("../Dataset/CASE_dataset/stats/min.pickle", "rb") as f:
    #    dmin = pickle.load(f)
##
    #dmeanf = defaultdict(di)
    #dmaxf = defaultdict(di)
    #dminf = defaultdict(di)
##
    #for subject in range(1, 31):
    #    for label in labels:
    #        dmeanf[subject][label] = np.mean(dmean[subject][label])
##
    #        max_cutoff = np.mean(dmax[subject][label]) + 3 * np.std(dmax[subject][label])
    #        dmaxf[subject][label] = np.max([mx for mx in dmax[subject][label] if mx <= max_cutoff]) - dmeanf[subject][label]
##
    #        min_cutoff = np.mean(dmin[subject][label]) - 3 * np.std(dmin[subject][label])
    #        dminf[subject][label] = np.min([mn for mn in dmin[subject][label] if mn >= min_cutoff]) - dmeanf[subject][label]
##
    #with open("../Dataset/CASE_dataset/stats/subj_mean.pickle", "wb") as f:
    #    pickle.dump(dmeanf, f)
    #with open("../Dataset/CASE_dataset/stats/subj_max.pickle", "wb") as f:
    #    pickle.dump(dmaxf, f)
    #with open("../Dataset/CASE_dataset/stats/subj_min.pickle", "wb") as f:
    #    pickle.dump(dminf, f)
#
    #with open("../Dataset/CASE_dataset/stats/subj_mean.pickle", "rb") as f:
    #   dmean = pickle.load(f)
    #with open("../Dataset/CASE_dataset/stats/subj_max.pickle", "rb") as f:
    #   dmax = pickle.load(f)
    #with open("../Dataset/CASE_dataset/stats/subj_min.pickle", "rb") as f:
    #   dmin = pickle.load(f)
    ##
    #dabsmax = defaultdict(di)
    ##
    #for subject in range(1, 31):
    #   for label in labels:
    #       dabsmax[subject][label] = max(np.abs(dmax[subject][label]), np.abs(dmin[subject][label]))
    ##
    ##
    #with open("../Dataset/CASE_dataset/stats/subj_minmax.pickle", "wb") as f:
    #   pickle.dump(dabsmax, f)

    #dmean = defaultdict(dl)
    #dmin = defaultdict(dl)
    #dmax = defaultdict(dl)
##
    #for data,__, subject in d:
    #    s = int(subject.numpy()[0])
    #    data = tf.squeeze(data)
    #    s_data = tf.split(data, num_or_size_splits=5, axis=0)
    #    mean = tf.reduce_mean(s_data, axis=1).numpy()
    #    min = tf.reduce_min(s_data, axis=1).numpy()
    #    max = tf.reduce_max(s_data, axis=1).numpy()
#
    #    for i, l in enumerate(labels):
    #        dmean[s][l].extend(mean[:, i])
    #        dmin[s][l].extend(min[:, i])
    #        dmax[s][l].extend(max[:, i])
#
    #with open("../Dataset/CASE_dataset/stats/mean.pickle", "wb") as f:
    #    pickle.dump(dmean, f)
    #with open("../Dataset/CASE_dataset/stats/max.pickle", "wb") as f:
    #    pickle.dump(dmax, f)
    #with open("../Dataset/CASE_dataset/stats/min.pickle", "wb") as f:
    #    pickle.dump(dmin, f)
