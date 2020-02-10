import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob
from scipy.signal import decimate

import seaborn as sns
import matplotlib.pyplot as plt

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def _float_feature(value, downsampling_rate):
    #assert len(value) == 5000
    value = decimate(value, q=downsampling_rate)
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def tfrecord_writer(data_path, window_size, stride, downsampling_rate=1, skip_baselines=True):

    os.chdir(data_path)
    for file in glob.glob("*.csv"):
        data = pd.read_csv(file)
        data = [pd.DataFrame(y) for x, y in data.groupby('video', as_index=False)]

        filename = os.path.basename(file).split(".")[0]
        print(filename)
        writer = tf.io.TFRecordWriter(f"../tfrecord_{window_size}d/{filename}.tfrecord")

        for video_data in data:
            if skip_baselines and video_data['video'].iloc[0] >= 10:
                continue

            size = len(video_data)
            for window_start in range(size % stride, size - window_size + 1, stride):
                features = tf.train.Features(
                    feature={
                        'Subject': _int64_feature(filename.split("_")[-1]),
                        'VideoID': _int64_feature(video_data['video'].iloc[0]),
                        'ecg': _float_feature(video_data['ecg'][window_start:window_start+window_size], downsampling_rate),
                        'bvp': _float_feature(video_data['bvp'][window_start:window_start+window_size], downsampling_rate),
                        'gsr': _float_feature(video_data['gsr'][window_start:window_start+window_size], downsampling_rate),
                        'rsp': _float_feature(video_data['rsp'][window_start:window_start+window_size], downsampling_rate),
                        'skt': _float_feature(video_data['skt'][window_start:window_start+window_size], downsampling_rate),
                        'valence': _float_feature(video_data['Valence'][window_start:window_start+window_size], downsampling_rate),
                        'arousal': _float_feature(video_data['Arousal'][window_start:window_start+window_size], downsampling_rate)
                    }
                )

                sequence_example = tf.train.Example(features=features)
                writer.write(sequence_example.SerializeToString())

        writer.close()

        print("yes")


def statistics_writer(data_path, file_ids):
    os.chdir(data_path)
    print(os.curdir)
    print(file_ids)
    dataset_mean, dataset_var = None, None
    for file in [glob.glob(f"sub_{num}.csv") for num in file_ids]:
        print(file[0])
        data = pd.read_csv(file[0])

        file_mean = data.mean(axis=0)
        file_var = data.var(axis=0)
        file_len = len(data)

        if dataset_mean is None:
            dataset_mean = file_mean
            dataset_var = file_var
            dataset_len = file_len
            continue

        summed_len = dataset_len + file_len
        # Welford's online algorithm
        for label, value in file_mean.items():
            delta = file_mean[label] - dataset_mean[label]
            m_data = dataset_var[label] * (dataset_len - 1)
            m_file = file_var[label] * (file_len - 1)
            dataset_m2 = m_data + m_file + delta ** 2 * dataset_len * file_len / summed_len
            dataset_var[label] = dataset_m2 / (summed_len -1)
            dataset_mean[label] = (dataset_len * dataset_mean[label] + file_len * file_mean[label]) / summed_len

        dataset_len = summed_len

    dataset_mean.to_csv(f"../stats/mean.csv", header=False)
    dataset_var.to_csv(f"../stats/var.csv", header=False)

def butter_lowpass_filter(data, cutoff=50, order=5):
    y = data
    y = decimate(y, q=10)
    print(y.dtype)
    return y

def downsampler(data_path):
    os.chdir(data_path)
    for file in glob.glob(f"*_13.csv"):
        data = pd.read_csv(file)

        sample = data['bvp'][234256:234256+15000]
        subsample = butter_lowpass_filter(sample)
        #subsample = decimate(sample, q=10)
        fig, (ax0, ax1) = plt.subplots(2, 1)
        ax0.plot(range(15000), sample)
        ax1.plot(range(1500), subsample)
        plt.show()
        break

if __name__ == '__main__':
    tfrecord_writer("../../Dataset/CASE_dataset/merged/", window_size=5000, stride=500, downsampling_rate=10)
    #statistics_writer("../../Dataset/CASE_dataset/merged/", range(1, 27))
    #downsampler("../../Dataset/CASE_dataset/merged/")