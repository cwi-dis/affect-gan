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
from models.DeepCNN2 import DeepCNN
from models.LateFuseCNN import LateFuseCNN
from models.AttentionNET import AttentionNET
from models.AttentionNET2 import AttentionNET as AttentionNET2
from models.AttentionNETDual import AttentionNETDual
from data.dataloader import Dataloader

from util.callbacks import CallbacksProducer
from util.CustomLosses import CastingBinaryCrossentropy
from util.CustomMetrics import *
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


def run(model_name, hparams, logdir, run_name=None, dense_shape=None):
    try:
        if hparams[config.HP_LOSS_TYPE] == "MSE":
            continuous_labels = True
            loss = tf.keras.losses.MeanSquaredError()
            metrics = [SimpleRegressionAccuracy]
            labels = ["arousal"]
        elif hparams[config.HP_LOSS_TYPE] == "BCE":
            continuous_labels = True
            loss = CastingBinaryCrossentropy()
            metrics = [CastingBinaryAccuracy()]
            labels = ["arousal"]
        elif hparams[config.HP_LOSS_TYPE] == "DUAL_BCE":
            continuous_labels = True
            loss = [CastingBinaryCrossentropy(), CastingBinaryCrossentropy()]
            metrics = [CastingBinaryAccuracy(), CastingBinaryAccuracy()]
            labels = ["arousal", "valence"]
    except:
        continuous_labels = True
        loss = CastingBinaryCrossentropy()
        metrics = [CastingBinaryAccuracy()]
        labels = ["arousal"]

    dataloader = Dataloader("5000d", ["bvp", "ecg", "rsp", "gsr", "skt"], labels, continuous_labels=continuous_labels)
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
    if model_name == "LateFuseCNN":
        model = LateFuseCNN(hparams, 5)
    if model_name == "AttentionNET":
        model = AttentionNET(hparams)
    if model_name == "AttentionNET2":
        model = AttentionNET2(hparams)
    if model_name == "AttentionNETDual":
        model = AttentionNETDual(hparams)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(clipnorm=1, learning_rate=0.0005),
        loss=loss,
        metrics=metrics
    )

    callback_dataset = dataloader("test_eval")
    callbacks = CallbacksProducer(hparams, logdir, run_name, callback_dataset).get_callbacks()

    model.fit(train_dataset, epochs=30, validation_data=eval_dataset, callbacks=callbacks)


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

                run(model_name, hparams, run_logdir, run_name)
                session_num += 1

    if model_name == "DeepCNN":
        for layers in config.HP_DEEP_LAYERS.domain.values:
            for upchannels in config.HP_DEEP_CHANNELS.domain.values:
                for ksize in config.HP_DEEP_KERNEL_SIZE.domain.values:
                    for loss in config.HP_LOSS_TYPE.domain.values:
                        for r in range(config.RUNS):
                            hparams = {
                                config.HP_DEEP_LAYERS: layers,
                                config.HP_DEEP_CHANNELS: upchannels,
                                config.HP_DEEP_KERNEL_SIZE: ksize,
                                config.HP_LOSS_TYPE: loss
                            }

                            run_name = "run-%d" % session_num
                            run_logdir = os.path.join(logdir, run_name)
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hparams[h] for h in hparams})
                            print("--- Restart %d of %d" % (r, config.RUNS))

                            run(model_name, hparams, run_logdir, run_name)
                            session_num += 1

    if model_name == "LateFuseCNN":
        for v_layers in config.HP_LDEEP_V_LAYERS.domain.values:
            for v_channels in config.HP_LDEEP_V_CHANNELS.domain.values:
                for f_layers in config.HP_LDEEP_F_LAYERS.domain.values:
                    for dropout in config.HP_LDEEP_DROPOUT.domain.values:
                        for ksize in config.HP_LDEEP_KSIZE.domain.values:
                            for wnorm in config.HP_LDEEP_WEIGHTNORM.domain.values:
                                hparams = {
                                    config.HP_LDEEP_V_LAYERS: v_layers,
                                    config.HP_LDEEP_V_CHANNELS: v_channels,
                                    config.HP_LDEEP_F_LAYERS: f_layers,
                                    config.HP_LDEEP_DROPOUT: dropout,
                                    config.HP_LDEEP_KSIZE: ksize,
                                    config.HP_LDEEP_WEIGHTNORM: wnorm
                                }

                                run_name = "run-%d" % session_num
                                run_logdir = os.path.join(logdir, run_name)
                                print('--- Starting trial: %s' % run_name)
                                print({h.name: hparams[h] for h in hparams})

                                run(model_name, hparams, run_logdir, run_name)
                                session_num += 1

    if model_name == "AttentionNET":
        for filters in config.HP_ATT_FILTERS.domain.values:
            for extra in config.HP_ATT_EXTRA_LAYER.domain.values:
                for attd in config.HP_ATT_DOWNRESATT.domain.values:
                    for upchannel in config.HP_ATT_UPCHANNEL.domain.values:
                        for r in range(config.RUNS):
                            hparams = {
                                config.HP_ATT_FILTERS: filters,
                                config.HP_ATT_EXTRA_LAYER: extra,
                                config.HP_ATT_DOWNRESATT: attd,
                                config.HP_ATT_UPCHANNEL: upchannel
                            }

                            run_name = "run-%d.%d" % (session_num, r)
                            run_logdir = os.path.join(logdir, run_name)
                            print('--- Starting trial: %s' % run_name)
                            print({h.name: hparams[h] for h in hparams})
                            print("--- Restart %d of %d" % (r+1, config.RUNS))
                            run(model_name, hparams, run_logdir, run_name)
                            session_num += 1

    if model_name == "AttentionNET2":
        for filters in config.HP_ATT2_FILTERS.domain.values:
            for layers in config.HP_ATT2_LAYERS.domain.values:
                for r in range(config.RUNS):
                    hparams = {
                        config.HP_ATT2_FILTERS: filters,
                        config.HP_ATT2_LAYERS: layers,
                    }

                    run_name = "run-%d.%d" % (session_num, r)
                    run_logdir = os.path.join(logdir, run_name)
                    print('--- Starting trial: %s' % run_name)
                    print({h.name: hparams[h] for h in hparams})
                    print("--- Restart %d of %d" % (r+1, config.RUNS))
                    run(model_name, hparams, run_logdir, run_name)
                session_num += 1

    if model_name == "AttentionNETDual":
        for filters in config.HP_ATTD_FILTERS.domain.values:
            for layers in config.HP_ATTD_LAYERS.domain.values:
                for comm in config.HP_ATTD_COMMON_INIT.domain.values:
                    for r in range(config.RUNS):
                        hparams = {
                            config.HP_ATTD_FILTERS: filters,
                            config.HP_ATTD_LAYERS: layers,
                            config.HP_ATTD_COMMON_INIT: comm,
                            config.HP_LOSS_TYPE: "DUAL_BCE"
                        }

                        run_name = "run-%d.%d" % (session_num, r)
                        run_logdir = os.path.join(logdir, run_name)
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        print("--- Restart %d of %d" % (r+1, config.RUNS))
                        run(model_name, hparams, run_logdir, run_name)
                    session_num += 1

def single_run(model_name):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("../Logs", model_name + run_id)
    hparams = config.OPT_PARAMS[model_name]
    run(model_name, hparams, logdir)


def main():
    init_tf_gpus()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("../Logs", run_id)

    single_run(model_name="AttentionNET2")
    #hp_sweep_run(logdir, model_name="AttentionNET")


def summary():
    hparams = config.OPT_PARAMS["AttentionNETDual"]
    #hparams = {
    #    config.HP_DEEP_LAYERS: 4,
    #    config.HP_DEEP_CHANNELS: 2,
    #    config.HP_DEEP_KERNEL_SIZE: 3,
    #    config.HP_LOSS_TYPE: "BCE"
    #}

    #ResNET(num_classes=1).model().summary()
    #SimpleLSTM(hparams).model().summary()
    #BaseNET1(hparams).model().summary()
    #ConvLSTM(hparams).model().summary()
    #ChannelCNN(hparams, 5).model().summary()
    #DeepCNN(hparams).model().summary()
    #LateFuseCNN(hparams, 5).model().summary()
    AttentionNETDual(hparams).model().summary()


if __name__ == '__main__':
    summary()
    #main()
