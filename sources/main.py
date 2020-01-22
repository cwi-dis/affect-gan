from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.SimpleLSTM import SimpleLSTM
from data.dataloader import Dataloader

def main():
    dataloader = Dataloader("../Dataset/CASE_dataset/tfrecord/", 1000, ["ecg", "rsp"], "arousal")

    model = SimpleLSTM()
    model.compile(
        optimizer='adam',
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.1),
        metrics=['accuracy']
    )

    train_dataset = dataloader(64, "train")
    eval_dataset = dataloader(64, "eval")

    tensorboard_cbk = tf.keras.callbacks.TensorBoard(
        log_dir='../Logs',
        histogram_freq=1,
        write_graph=True,
        update_freq=100
    )
    model.fit(train_dataset, epochs=2, validation_data=eval_dataset, callbacks=[tensorboard_cbk])


if __name__ == '__main__':
    main()
