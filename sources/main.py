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
from models.TAGAN import Generator, Discriminator
from data.dataloader import Dataloader
from data.datagenerator import DatasetGenerator
from train_gan import GAN_Trainer

from util.callbacks import CallbacksProducer
from util.misc import init_tf_gpus
from util.CustomLosses import CastingBinaryCrossentropy
from util.CustomMetrics import *
from tensorboard.plugins.hparams import api as hp


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
        optimizer=tf.keras.optimizers.Adam(clipnorm=1, learning_rate=0.0008),
        loss=loss,
        metrics=metrics
    )

    callback_dataset = dataloader("test_eval")
    callbacks = CallbacksProducer(hparams, logdir, run_name, callback_dataset).get_callbacks()

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

                                # dense_shape = tf.math.ceil(config.INPUT_SIZE / pool) * filters
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
                            print("--- Restart %d of %d" % (r + 1, config.RUNS))
                            run(model_name, hparams, run_logdir, run_name)
                            session_num += 1

    if model_name == "AttentionNET2":
        for filters in config.HP_ATT2_FILTERS.domain.values:
            for late_filters in config.HP_ATT2_OTHERFILTERS.domain.values:
                for layers in config.HP_ATT2_LAYERS.domain.values:
                    for r in range(config.RUNS):
                        hparams = {
                            config.HP_ATT2_FILTERS: filters,
                            config.HP_ATT2_LAYERS: layers,
                            config.HP_ATT2_OTHERFILTERS: late_filters
                        }

                        run_name = "run-%d.%d" % (session_num, r)
                        run_logdir = os.path.join(logdir, run_name)
                        print('--- Starting trial: %s' % run_name)
                        print({h.name: hparams[h] for h in hparams})
                        print("--- Restart %d of %d" % (r + 1, config.RUNS))
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
                        print("--- Restart %d of %d" % (r + 1, config.RUNS))
                        run(model_name, hparams, run_logdir, run_name)
                    session_num += 1


def single_run(model_name):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("../Logs", model_name + run_id)
    hparams = config.OPT_PARAMS[model_name]
    run(model_name, hparams, logdir)


def run_gan(model_name):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("../Logs", model_name + run_id)
    leave_out = 3
    hparams = config.OPT_PARAMS["gan"]
    labels = config.LABELS
    dataloader = Dataloader(
        "5000d", labels, continuous_labels=False,
        normalized=True
    )
    dataset = dataloader("gan", hparams[config.HP_GAN_BATCHSIZE], leave_out=leave_out)
    trainer = GAN_Trainer(
        mode=model_name,
        batch_size=hparams[config.HP_GAN_BATCHSIZE],
        hparams=hparams,
        logdir=logdir,
        num_classes=2,
        n_signals=len(labels),
        leave_out=leave_out,
        class_conditional=True,
        subject_conditional=True,
        save_image_every_n_steps=1500,
        n_critic=5,
        train_steps=200000
    )

    trainer.train(dataset=dataset)


def train_loso_gans(model_name):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("../Logs", "loso-" + model_name + run_id)
    hparams = config.OPT_PARAMS["gan"]

    dataloader = Dataloader(
        "5000d", ["ecg"],
        normalized=True,
        continuous_labels=False
    )

    for out_subject in config.OUT_SUBJECT.domain.values:
        run_name = "subject-%d-out" % out_subject
        run_logdir = os.path.join(logdir, run_name)
        tf.print("Training GAN sans subject %d." % out_subject)
        dataset = dataloader("gan", hparams[config.HP_GAN_BATCHSIZE], leave_out=out_subject)
        trainer = GAN_Trainer(
            mode=model_name,
            batch_size=hparams[config.HP_GAN_BATCHSIZE],
            hparams=hparams,
            logdir=run_logdir,
            num_classes=2,
            n_signals=1,
            leave_out=out_subject,
            class_conditional=True,
            subject_conditional=False,
            save_image_every_n_steps=1500,
            n_critic=5,
            train_steps=150000
        )

        trainer.train(dataset=dataset)

        del dataset
        del trainer


def run_loso_cv(model_name):
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("../Logs", "loso-" + model_name + run_id)
    dataloader = Dataloader(
        "5000d", ["ecg"],
        normalized=True,
        continuous_labels=False
    )
    generator_base_path = "../Logs"

    for out_subject in config.OUT_SUBJECT.domain.values:
        subject_label = "subject-%d-out" % out_subject
        eval_set = dataloader(mode="eval", batch_size=128, leave_out=out_subject)

        for data_source in config.TRAIN_DATA.domain.values:
            hparams = {config.OUT_SUBJECT: out_subject,
                       config.TRAIN_DATA: data_source}
            run_name = "%s-%s" % (subject_label, data_source)
            if data_source is "real":
                train_set = dataloader(mode="train", batch_size=128, leave_out=out_subject)
            else:
                train_label = data_source.split('_')
                wgan_path = "loso-wgan-class" if train_label[1] is "cls" else "loso-wgan-class-subject"
                wgan_path = os.path.join(generator_base_path, wgan_path, subject_label, "model_gen")
                subj_cond = True if train_label[1] is "subjcls" else False
                categorical_sampling = True if train_label[2] is "catg" else False
                train_set = DatasetGenerator(batch_size=128,
                                             generator_path=wgan_path,
                                             class_conditioned=True,
                                             subject_conditioned=subj_cond,
                                             categorical_sampling=categorical_sampling,
                                             no_subject_output=True).__call__()
            for rerun in range(config.NUM_RERUNS):
                print("Subject: %d, Trained on %s data, Restart #%d" % (out_subject, data_source, rerun))
                run_logdir = os.path.join(logdir, run_name, ".%d" % rerun)

                if model_name == "AttentionNET":
                    model = AttentionNET(hparams)

                model.compile(
                    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0008, beta_1=0.5, beta_2=0.9),
                    loss=[tf.keras.losses.KLD],
                    metrics=["accuracy"]
                )

                callbacks = CallbacksProducer(hparams, run_logdir, run_name).get_callbacks()

                model.fit(train_set, epochs=200, steps_per_epoch=50, validation_data=eval_set, callbacks=callbacks)

                del model

            del train_set


def main():
    init_tf_gpus()

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S")
    logdir = os.path.join("../Logs", run_id)

    # single_run(model_name="AttentionNET2")
    # run_loso_cv(model_name="AttentionNET")
    run_gan(model_name="wgan-gp")
    # train_loso_gans(model_name="wgan-gp")
    # hp_sweep_run(logdir, model_name="AttentionNET")


def summary():
    hparams = config.OPT_PARAMS["gan"]
    # hparams = {
    #    config.HP_DEEP_LAYERS: 4,
    #    config.HP_DEEP_CHANNELS: 2,
    #    config.HP_DEEP_KERNEL_SIZE: 3,
    #    config.HP_LOSS_TYPE: "BCE"
    # }

    # ResNET(num_classes=1).model().summary()
    # SimpleLSTM(hparams).model().summary()
    # BaseNET1(hparams).model().summary()
    # ConvLSTM(hparams).model().summary()
    # ChannelCNN(hparams, 5).model().summary()
    # DeepCNN(hparams).model().summary()
    # LateFuseCNN(hparams, 5).model().summary()
    # AttentionNET(hparams).model().summary()
    Generator(n_signals=5).model().summary()
    Discriminator(conditional=True).model().summary()


if __name__ == '__main__':
    # summary()
    main()
