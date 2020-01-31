import tensorflow as tf
import numpy as np
import pandas as pd
import os
import glob

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))

def _float_feature(value):
    #assert len(value) == 5000
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def tfrecord_writer(data_path, window_size, stride):

    os.chdir(data_path)
    for file in glob.glob("*.csv"):
        data = pd.read_csv(file)
        data = [pd.DataFrame(y) for x, y in data.groupby('video', as_index=False)]

        filename = os.path.basename(file).split(".")[0]
        print(filename)
        writer = tf.io.TFRecordWriter(f"../tfrecord_{window_size}/{filename}.tfrecord")

        for video_data in data:
            size = len(video_data)
            for window_start in range(size % stride, size - window_size + 1, stride):
                features = tf.train.Features(
                    feature={
                        'Subject': _int64_feature(filename.split("_")[-1]),
                        'VideoID': _int64_feature(video_data['video'].iloc[0]),
                        'ecg': _float_feature(video_data['ecg'][window_start:window_start+window_size]),
                        'bvp': _float_feature(video_data['bvp'][window_start:window_start+window_size]),
                        'gsr': _float_feature(video_data['gsr'][window_start:window_start+window_size]),
                        'rsp': _float_feature(video_data['rsp'][window_start:window_start+window_size]),
                        'skt': _float_feature(video_data['skt'][window_start:window_start+window_size]),
                        'valence': _float_feature(video_data['Valence'][window_start:window_start+window_size]),
                        'arousal': _float_feature(video_data['Arousal'][window_start:window_start+window_size])
                    }
                )

                sequence_example = tf.train.Example(features=features)
                writer.write(sequence_example.SerializeToString())

        writer.close()

        print("yes")



if __name__ == '__main__':
    tfrecord_writer("../../Dataset/CASE_dataset/merged/", window_size=5000, stride=1000)