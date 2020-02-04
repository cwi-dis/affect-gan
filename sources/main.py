from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from models.ResNET import ResNET
from data.dataloader import Dataloader


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

def init_callbacks():
    callbacks = []

    run_id = tf.timestamp()

    callbacks.append(tf.keras.callbacks.TensorBoard(
        log_dir=f"../Logs/Resnet/{run_id}/",
        histogram_freq=1,
        write_graph=True,
        update_freq=100
    ))

    callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
        monitor='loss',
        factor=0.5,
        patience=5,
        min_lr=0.0001
    ))

    callbacks.append(tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=10
    ))

    return callbacks

def main():
    init_tf_gpus()

    dataloader = Dataloader("../Dataset/CASE_dataset/tfrecord_3000/", ["bvp", "ecg"], ["arousal"])

    model = ResNET(num_classes=1)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    train_dataset = dataloader(64, "train")
    eval_dataset = dataloader(150, "eval")

    callbacks = init_callbacks()

    model.fit(train_dataset, epochs=100, validation_data=eval_dataset, validation_steps=15, callbacks=callbacks)


if __name__ == '__main__':

    main()