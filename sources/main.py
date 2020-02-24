from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from datetime import datetime
import config

from models.BaseNET1 import BaseNET1
from models.SimpleLSTM import SimpleLSTM
from data.dataloader import Dataloader

from util.callbacks import CallbacksProducer
from tensorboard.plugins.hparams import api as hp


def init_tf_gpus():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("Physical GPUs available: %d  -- Logical GPUs available: %d" % (len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            print(e)

def run(model_name, hparams, logdir, run_name, dense_shape):

    dataloader = Dataloader("5000d", ["bvp", "ecg", "rsp", "gsr", "skt"], ["arousal"])
    train_dataset = dataloader("train", 128)
    eval_dataset = dataloader("eval", 128)

    if model_name == "BaseNET":
        model = BaseNET1(hparams, dense_shape)  # ResNET(num_classes=1)
    if model_name == "SimpleLSTM":
        model = SimpleLSTM(hparams)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1, learning_rate=hparams[config.HP_LR]),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        metrics=['accuracy']
    )

    callbacks = CallbacksProducer(hparams, logdir, run_name).get_callbacks()

    model.fit(train_dataset, epochs=30, validation_data=eval_dataset, validation_steps=45, callbacks=callbacks)

def hp_sweep_run(logdir, run_id, model_name):
    session_num = 0

    if model_name == "BaseNET":
        for filters in config.HP_FILTERS.domain.values:
            for dropout in config.HP_DROPOUT.domain.values:
                for kernel_size in config.HP_KERNEL.domain.values:
                    for dilation in config.HP_DILATION.domain.values:
                        for pool in config.HP_POOL.domain.values:
                            for lr in config.HP_LR.domain.values:
                                hparams = {
                                    config.HP_FILTERS: filters,
                                    config.HP_DROPOUT: dropout,
                                    config.HP_KERNEL: kernel_size,
                                    config.HP_DILATION: dilation,
                                    config.HP_POOL: pool,
                                    config.HP_LR: lr
                                }

                                dense_shape = tf.math.ceil(config.INPUT_SIZE / pool) * filters
                                run_name = "run-%d" % session_num
                                run_logdir = os.path.join(logdir, run_name)
                                print('--- Starting trial: %s' % run_name)
                                print({h.name: hparams[h] for h in hparams})

                                run(hparams, run_logdir, run_name, dense_shape)
                                session_num += 1

    if model_name == "SimpleLSTM":
        for cells in config.HP_CELLS_LSTM.domain.values:
            for lr in config.HP_LR_LSTM.domain.values:
                hparams = {
                    config.HP_CELLS_LSTM: cells,
                    config.HP_LR_LSTM: lr
                }

                run_name = "run-%d" % session_num
                run_logdir = os.path.join(logdir, run_name)
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})

                run(hparams, run_logdir, run_name)
                session_num += 1

def main():
    init_tf_gpus()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("../Logs", run_id)

    hp_sweep_run(logdir, run_id, model_name="BaseNET")



def summary():

    hparams = {
        config.HP_FILTERS: 8,
        config.HP_DROPOUT: 0.5,
        config.HP_KERNEL: 3,
        config.HP_DILATION: 2,
        config.HP_POOL: 3,
        config.HP_LR: 0.001,
        "dense_shape": tf.math.ceil(500 / 3) * 8
    }

    #ResNET(num_classes=1).model().summary()
    #SimpleLSTM().model().summary()
    BaseNET1(hparams).model().summary()


if __name__ == '__main__':
    #summary()
    main()
