from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.ResNET import ResNET
from models.BaseNET1 import BaseNET1
from models.SimpleLSTM import SimpleLSTM
from data.dataloader import Dataloader

from util.callbacks import CallbacksProducer


def init_tf_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(f"Physical GPUs available: {len(gpus)} -- Logical GPUs available: {len(logical_gpus)}")
        except RuntimeError as e:
            print(e)


def main():
    init_tf_gpus()

    dataloader = Dataloader("5000d", ["bvp", "ecg", "rsp", "gsr", "skt"], ["arousal"])

    model = BaseNET1()#ResNET(num_classes=1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    train_dataset = dataloader("train", 128)
    eval_dataset = dataloader("eval", 128)

    callbacks = CallbacksProducer().get_callbacks()

    model.fit(train_dataset, epochs=50, validation_data=eval_dataset, validation_steps=26, callbacks=callbacks)


def summary():
    #ResNET(num_classes=1).model().summary()
    #SimpleLSTM().model().summary()
    BaseNET1().model().summary()


if __name__ == '__main__':
    summary()
    #main()
