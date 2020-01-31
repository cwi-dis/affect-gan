import os
import glob
import tensorflow as tf



def window_dataset(dataset):
    window_ds = dataset.window(1000, shift=1000)

    def sub_to_batch(sub):
        return sub.batch(1000, drop_remainder=True)

    windows = window_ds.flat_map(sub_to_batch)
    return windows


def with_categoric_labels(features, label):
    return features, 0 if label > 5 else 1


class Dataloader(object):

    def __init__(self, path="/root/UvA/thesis/affect-gan/Dataset/CASE_dataset/tfrecord_5000/", features=None, label=None):
        if features is None:
            features = ["bvp"]
        if path is None or not os.path.exists(path):
            raise Exception(f"Data path does not exist: {os.curdir}:{path}")

        self.path = path
        self.next_element = None
        #self.window_size = window_size
        self.features = features
        if label is None:
            self.labels = ["arousal"]
        else:
            self.labels = label

        self.excluded_label = tf.constant(5, tf.float32)
        self.train_num = range(1, 27)
        self.eval_num = [27]
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

        subject = tf.cast(decoded_example["Subject"], tf.int8)

        features = []
        for feature in self.features:
            features.append(decoded_example[feature])

        labels = []
        for label in self.labels:
            value = tf.reduce_mean(decoded_example[label][-50:])
            labels.append(value)


        features = tf.stack(features, axis=1)
        label = tf.stack(labels)

        return features, label

    def __call__(self, batch_size, mode, leave_out=None):

        modes = ["train", "test", "eval", "inspect"]
        if mode not in modes:
            raise "mode not found! supported modes are " + modes
        #if mode is "eval" and leave_out is None:
        #    raise Exception("leave-one-out evaluation undefined!")

        if mode is "train":
            files = [glob.glob(f"{self.path}sub_{num}.tfrecord") for num in self.train_num]
        elif mode is "eval":
            files = [glob.glob(f"{self.path}sub_{num}.tfrecord") for num in self.eval_num]
        elif mode is "test":
            files = [glob.glob(f"{self.path}sub_{num}.tfrecord") for num in self.test_num]
        elif mode is "inspect":
            files = [glob.glob(f"{self.path}*.tfrecord")]
            print(f"Files loaded in mode {mode}: {files}")

            dataset = tf.data.TFRecordDataset(files)
            dataset = dataset.map(self._decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            dataset = dataset.filter(lambda _, label: tf.reduce_any(tf.greater(tf.abs(label - self.excluded_label), 0.05)))
            return dataset

        print(f"Files loaded in mode {mode}: {files}")
        files = tf.data.Dataset.from_tensor_slices(files)

        dataset = files.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.map(self._decode, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        dataset = dataset.filter(lambda _, label: tf.reduce_any(tf.greater(tf.abs(label - self.excluded_label), 0.05)))
        dataset = dataset.map(with_categoric_labels)

        if mode == "train":
            dataset = dataset.shuffle(buffer_size=1000)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)

        return dataset


if __name__ == '__main__':
    d = Dataloader("../../Dataset/CASE_dataset/tfrecord_5000/", ["ecg", "rsp"], ["arousal", "valence"])
    d = d(1, "inspect")

    total_count = 0
    good = 0
    for element, label in d:
        print(element)

    print(good)