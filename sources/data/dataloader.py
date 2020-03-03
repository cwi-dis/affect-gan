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

    def __init__(self, datasetID, features=None, label=None, continuous_labels=False, normalized=True):
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

        self.means = pd.read_csv("../Dataset/CASE_dataset/stats/mean.csv", header=None, index_col=0, squeeze=True).astype("float32")
        self.vars = pd.read_csv("../Dataset/CASE_dataset/stats/var.csv", header=None, index_col=0, squeeze=True).astype("float32")

        self.excluded_label = tf.constant(5, tf.float32)
        self.train_num = range(1, 27)
        self.eval_num = [27, 28, 29, 30]
        self.test_num = range(28, 31)

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
                features.append((decoded_example[feature] - self.means[feature]) / tf.sqrt(self.vars[feature]))
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

        modes = ["train", "test", "eval", "inspect"]
        if mode not in modes:
            raise Exception("mode not found! supported modes are %s" % modes)
        #if mode is "eval" and leave_out is None:
        #    raise Exception("leave-one-out evaluation undefined!")

        if mode is "train":
            files = [glob.glob("%ssub_%d.tfrecord" % (self.path, num)) for num in self.train_num]
        elif mode is "eval":
            files = [glob.glob("%ssub_%d.tfrecord" % (self.path, num)) for num in self.eval_num]
        elif mode is "test":
            files = [glob.glob("%ssub_%d.tfrecord" % (self.path, num)) for num in self.test_num]
        elif mode is "inspect":
            files = [glob.glob("%s*.tfrecord" % self.path)]

        #print(f"Files loaded in mode %s: {files}")
        files = tf.data.Dataset.from_tensor_slices(files)

        dataset = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE, cycle_length=10, block_length=128)
        dataset = dataset.map(self._decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.filter(lambda _, __, video: tf.less(video, 10))
        dataset = dataset.map(lambda features, labels, video: (features, labels))
        dataset = dataset.filter(lambda _, label: tf.reduce_all(tf.greater(tf.abs(label - self.excluded_label), 0.25)))

        if mode is "inspect":
            dataset = dataset.shuffle(buffer_size=500)
            return dataset

        if not self.continuous_labels:
            dataset = dataset.map(with_categoric_labels)

        if len(self.labels) == 2:
            dataset = dataset.map(lambda data, labels: (data, (labels[0], labels[1])))

        if mode == "train":
            dataset = dataset.shuffle(buffer_size=2)
        
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        return dataset


if __name__ == '__main__':
    os.chdir("./..")
    d = Dataloader("5000d", ["ecg", "rsp"], ["arousal", "valence"])
    d = d("train", 1)

    i=0
    for _, label1 in d.take(1):
        print(label1)
    print(i)

