import os
import glob
import tensorflow as tf
import pandas as pd
from scipy.signal import decimate
import numpy as np

from tensorflow.python.ops import math_ops

def window_dataset(dataset):
    window_ds = dataset.window(1000, shift=1000)

    def sub_to_batch(sub):
        return sub.batch(1000, drop_remainder=True)

    windows = window_ds.flat_map(sub_to_batch)
    return windows


def with_categoric_labels(features, label, threshold=5.0):
    return features, math_ops.cast(label > threshold, label.dtype)


class Dataloader(object):

    def __init__(self, datasetID, features=None, label=None, continuous_labels=True, normalized=True, range_clipped=False):
        if features is None:
            features = ["bvp"]
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
        self.range_clipped = range_clipped

        self.means = pd.read_csv("../Dataset/CASE_dataset/stats/mean.csv", header=None, index_col=0, squeeze=True).astype("float32")
        self.vars = pd.read_csv("../Dataset/CASE_dataset/stats/var.csv", header=None, index_col=0, squeeze=True).astype("float32")
        self.minmax = pd.read_csv("../Dataset/CASE_dataset/stats/minmax.csv", header=0, index_col=0, squeeze=True).astype("float32")

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

        subject = tf.cast(decoded_example["Subject"], tf.float32)
        video = tf.cast(decoded_example["VideoID"], tf.float32)

        features = []
        if self.normalized:
            for feature in self.features:
                znorm = (decoded_example[feature] - self.means[feature]) / tf.sqrt(self.vars[feature])
                if self.range_clipped:
                    znorm = 2 * (znorm - self.minmax[min][feature]) / (self.minmax[max][feature] - self.minmax[min][feature]) - 1
                features.append(znorm)
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

        return features, labels, video

    def __call__(self, mode, batch_size=64, leave_out=None):

        modes = ["train", "test", "eval", "inspect", "test_eval", "gan", "cgan"]
        if mode not in modes:
            raise Exception("mode not found! supported modes are %s" % modes)
        #if mode is "eval" and leave_out is None:
        #    raise Exception("leave-one-out evaluation undefined!")

        if leave_out is None:
            train_subject_ids = self.subject_labels[:28]
            eval_subject_ids = self.subject_labels[28:]
        else:
            train_subject_ids = self.subject_labels[:leave_out] + self.subject_labels[leave_out+1:]
            eval_subject_ids = self.subject_labels[leave_out:leave_out+1]

        if (mode is "train") or (mode is "gan"):
            files = [glob.glob("%ssub_%d.tfrecord" % (self.path, num)) for num in train_subject_ids]
        elif (mode is "eval") or (mode is "test_eval"):
            files = [glob.glob("%ssub_%d.tfrecord" % (self.path, num)) for num in eval_subject_ids]
        else:
            files = [glob.glob("%s*.tfrecord" % self.path)]

        print(f"Files loaded in mode %s: {files}")
        files = tf.data.Dataset.from_tensor_slices(files)

        dataset = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE, cycle_length=25, block_length=128)
        dataset = dataset.map(self._decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.filter(lambda _, __, video: tf.less(video, 10))
        dataset = dataset.map(lambda features, labels, video: (features, labels))
        if mode is not ("gan" or "inspect"):
            dataset = dataset.filter(lambda _, label: tf.reduce_all(tf.greater(tf.abs(label - self.excluded_label), 0.2)))
        else:
            dataset = dataset.filter(lambda _, label: tf.reduce_all(tf.greater(tf.abs(label - self.excluded_label), 0.05)))
        if self.range_clipped:
            dataset = dataset.filter(lambda features, _: tf.less_equal(tf.reduce_max(features), 1) and tf.less_equal(tf.abs(tf.reduce_min(features)), 1))

        if not self.continuous_labels:
            dataset = dataset.map(with_categoric_labels)

        if mode is "inspect":
            dataset = dataset.shuffle(buffer_size=10000)
            return dataset

        if len(self.labels) == 2:
            dataset = dataset.map(lambda data, labels: (data, (labels[0], labels[1])))

        if mode == "train":
            dataset = dataset.shuffle(buffer_size=2)

        if mode == "gan":
            #dataset = dataset.map(lambda features, labels: features)
            dataset = dataset.shuffle(buffer_size=300)
            dataset = dataset.repeat()

        if mode == "cgan":
            dataset = dataset.shuffle(buffer_size=3)
            dataset = dataset.repeat()

        if mode == "test_eval":
            return dataset.shuffle(1000, seed=42).batch(500).take(1)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        return dataset


if __name__ == '__main__':
    os.chdir("./..")
    d = Dataloader("5000d", ["ecg", "bvp", "gsr", "skt", "rsp"], range_clipped=True)
    d = d("inspect", 1)

    for e in d.take(10):
        print(e[0])

    #min = None
    #max = None
#
    #for batch in d:
    #    if min is None:
    #        min = tf.reduce_min(batch[0], axis=0)
    #        max = tf.reduce_max(batch[0], axis=0)
    #    else:
    #        mn = tf.abs(tf.reduce_min(batch[0])) > 6
    #        mx = tf.reduce_max(batch[0]) > 6
    #        if not (mn or mx):
    #            min = tf.reduce_min(tf.concat([[min], [tf.reduce_min(batch[0], axis=0)]], axis=0), axis=0)
    #            max = tf.reduce_max(tf.concat([[max], [tf.reduce_max(batch[0], axis=0)]], axis=0), axis=0)
    #min = min.numpy()
    #max = max.numpy()
    #df = pd.DataFrame.from_dict(
    #    {
    #        "ecg": [min[0], max[0]],
    #        "bvp": [min[1], max[1]],
    #        "gsr": [min[2], max[2]],
    #        "skt": [min[3], max[3]],
    #        "rsp": [min[4], max[4]],
    #    }, orient="index", columns=["min", "max"])
#
    #df.to_csv(f"../Dataset/CASE_dataset/stats/minmax.csv" )
