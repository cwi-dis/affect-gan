from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from datetime import datetime
import config
import multiprocessing

from models.BaseNET1 import BaseNET1
from models.SimpleLSTM import SimpleLSTM
from models.ConvLSTM import ConvLSTM
from models.ChannelCNN import ChannelCNN
from models.DeepCNN import DeepCNN
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

def run(model_name, hparams, logdir, run_name, dense_shape=None):

    dataloader = Dataloader("5000d", ["bvp", "ecg", "rsp", "gsr", "skt"], ["arousal"])
    train_dataset = dataloader("train", 128)
    eval_dataset = dataloader("eval", 128)

    if model_name == "BaseNET":
        model = BaseNET1(hparams, dense_shape)  # ResNET(num_classes=1)
    if model_name == "SimpleLSTM":
        model = SimpleLSTM(hparams)
    if model_name == "ConvLSTM":
        model = ConvLSTM(hparams)
    if model_name == "ChannelCNN":
        model = ChannelCNN(hparams, 5)
    if model_name == "DeepCNN":
        model = DeepCNN(hparams)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1, learning_rate=0.0004),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05),
        metrics=['accuracy']
    )

    callbacks = CallbacksProducer(hparams, logdir, run_name).get_callbacks()

    model.fit(train_dataset, epochs=30, validation_data=eval_dataset, validation_steps=45, callbacks=callbacks)


def hp_sweep_run(logdir, model_name):
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

                                run(model_name, hparams, run_logdir, run_name, dense_shape)
                                session_num += 1

    if model_name == "SimpleLSTM":
        for cells in config.HP_CELLS_LSTM.domain.values:
            for lr in config.HP_LR_LSTM.domain.values:
                for in_drop in config.HP_IN_DROPOUT_LSTM.domain.values:
                    for rec_drop in config.HP_DROPOUT_LSTM.domain.values:
                        hparams = {
                            config.HP_CELLS_LSTM: cells,
                            config.HP_LR_LSTM: lr,
                            config.HP_IN_DROPOUT_LSTM: in_drop,
                            config.HP_DROPOUT_LSTM: rec_drop
                        }

                        run_name = "run-%d" % session_num
                        run_logdir = os.path.join(logdir, run_name)
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})

                        run(model_name, hparams, run_logdir, run_name)
                        session_num += 1

    if model_name == "ConvLSTM":
        for filters in config.HP_FILTERS_CL.domain.values:
            for dropout in config.HP_DROPOUT_CL.domain.values:
                for kernel_size in config.HP_KERNEL_CL.domain.values:
                    for strides in config.HP_STRIDES_CL.domain.values:
                        for cells in config.HP_LSTMCELLS_CL.domain.values:
                            for lr in config.HP_LR_CL.domain.values:
                                hparams = {
                                    config.HP_FILTERS_CL: filters,
                                    config.HP_DROPOUT_CL: dropout,
                                    config.HP_KERNEL_CL: kernel_size,
                                    config.HP_STRIDES_CL: strides,
                                    config.HP_LSTMCELLS_CL: cells,
                                    config.HP_LR_CL: lr
                                }

                                #dense_shape = tf.math.ceil(config.INPUT_SIZE / pool) * filters
                                run_name = "run-%d" % session_num
                                run_logdir = os.path.join(logdir, run_name)
                                print('--- Starting trial: %s' % run_name)
                                print({h.name: hparams[h] for h in hparams})

                                run(model_name, hparams, run_logdir, run_name)
                                session_num += 1

    if model_name == "ChannelCNN":
        for layers in config.HP_CDEEP_LAYERS.domain.values:
            for upchannels in config.HP_CDEEP_CHANNELS.domain.values:
                hparams = {
                    config.HP_CDEEP_LAYERS: layers,
                    config.HP_CDEEP_CHANNELS: upchannels
                }

                run_name = "run-%d" % session_num
                run_logdir = os.path.join(logdir, run_name)
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})

                process_train = multiprocessing.Process(target=run, args=(model_name, hparams, run_logdir, run_name,))
                process_train.start()
                process_train.join()
                run(model_name, hparams, run_logdir, run_name)
                session_num += 1

    if model_name == "DeepCNN":
        for layers in config.HP_DEEP_LAYERS.domain.values:
            for upchannels in config.HP_DEEP_CHANNELS.domain.values:
                hparams = {
                    config.HP_DEEP_LAYERS: layers,
                    config.HP_DEEP_CHANNELS: upchannels
                }

                run_name = "run-%d" % session_num
                run_logdir = os.path.join(logdir, run_name)
                print('--- Starting trial: %s' % run_name)
                print({h.name: hparams[h] for h in hparams})

                run(model_name, hparams, run_logdir, run_name)
                session_num += 1


def main():
    init_tf_gpus()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("../Logs", run_id)

    hp_sweep_run(logdir, model_name="ChannelCNN")


def summary():
    hparams = {
        config.HP_CDEEP_LAYERS: 4,
        config.HP_CDEEP_CHANNELS: 2,
    }

    #ResNET(num_classes=1).model().summary()
    #SimpleLSTM(hparams).model().summary()
    #BaseNET1(hparams).model().summary()
    #ConvLSTM(hparams).model().summary()
    ChannelCNN(hparams, 5).model().summary()
    #DeepCNN(hparams).model().summary()


if __name__ == '__main__':
    summary()
    #main()
