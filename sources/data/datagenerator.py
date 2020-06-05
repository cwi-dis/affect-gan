import tensorflow as tf
import numpy as np
import os
#from main import init_tf_gpus
#from models.TAGAN import *


class _DataGenerator:
    def __init__(self, file_path, batch_size, noise_dim, subject_conditioned, class_categorical_sampling, subject_categorical_sampling, discriminator_class_conditioned):
        self.gen_file_path = os.path.join(file_path, "model_gen")
        self.dis_file_path = os.path.join(file_path, "model_dis")
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.subject_conditioned = subject_conditioned
        self.class_categorical_sampling = class_categorical_sampling
        self.subject_categorical_sampling = subject_categorical_sampling
        self.discriminator_class_conditioned = discriminator_class_conditioned

        self.generator = tf.keras.models.load_model(
            filepath=self.gen_file_path
        )

        self.discriminator = tf.keras.models.load_model(
            filepath=self.dis_file_path
        )

        self.last_seed = None

    def __call__(self, *args, **kwargs):
        while True:
            seed = tf.random.normal(shape=[self.batch_size, self.noise_dim])
            if self.class_categorical_sampling:
                generator_class_inputs = tf.one_hot(tf.random.uniform([self.batch_size], maxval=2, dtype=tf.int32), depth=2)
            else:
                generator_class_inputs = tf.keras.activations.softmax(tf.random.normal([self.batch_size, 2], stddev=1))
            if self.subject_categorical_sampling:
                generator_subject_inputs = tf.one_hot(tf.random.uniform([self.batch_size], maxval=29, dtype=tf.int32), depth=29)
            else:
                generator_subject_inputs = tf.keras.activations.softmax(tf.random.normal([self.batch_size, 29], stddev=5))

            # arousal seed
            seed = tf.concat([seed, generator_class_inputs], axis=-1)

            if self.subject_conditioned:
                seed = tf.concat([seed, generator_subject_inputs], axis=-1)
            gen_sig = self.generator(seed, training=False)

            if self.discriminator_class_conditioned:
                _, _, generator_class_inputs, _ = self.discriminator(gen_sig)

            yield gen_sig, generator_class_inputs, generator_subject_inputs

    def get(self, arousal_seed, subject_seed, noise_seed_reuse):
        if noise_seed_reuse and self.last_seed is not None:
            seed = self.last_seed
        else:
            seed = tf.random.normal(shape=[1, self.noise_dim])
            self.last_seed = seed

        # arousal seed
        seed = tf.concat([seed, tf.cast(arousal_seed, tf.float32)], axis=-1)

        if self.subject_conditioned:
            seed = tf.concat([seed, tf.cast(subject_seed, tf.float32)], axis=-1)

        gen_sig = self.generator(seed, training=False)
        _, _, dis_class, _ = self.discriminator(gen_sig)

        return gen_sig, dis_class


class DatasetGenerator:
    def __init__(self, batch_size, path, subject_conditioned, class_categorical_sampling, subject_categorical_sampling, discriminator_class_conditioned, no_subject_output=False, argmaxed_label=False):
        self.batch_size = batch_size
        self.no_subject_output = no_subject_output
        self.argmaxed_label = argmaxed_label

        noise_dim = 100 if subject_conditioned else 129
        self.generator = _DataGenerator(path, batch_size, noise_dim, subject_conditioned, class_categorical_sampling, subject_categorical_sampling, discriminator_class_conditioned)

    def __call__(self, *args, **kwargs):
        datagenerator = tf.data.Dataset.from_generator(
            generator= self.generator,
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=((self.batch_size, 500, None), (self.batch_size, 2), (self.batch_size, 29))
        )

        if self.argmaxed_label:
            datagenerator = datagenerator.map(lambda signal, label, subject: (signal, tf.one_hot(tf.argmax(label, axis=-1), depth=2), subject))

        if self.no_subject_output:
            datagenerator = datagenerator.map(lambda signal, label, subject: (signal, label))

        return datagenerator

    def get(self, arousal_value, subject_value, sub0, sub1, noise_seed_reuse=False):
        arousal_seed = tf.expand_dims([arousal_value, 1-arousal_value], axis=0)
        subject_seed = np.zeros(29)
        subject_seed[sub0] = subject_value
        subject_seed[sub1] = 1 - subject_value
        subject_seed = tf.expand_dims(subject_seed, axis=0)

        return self.generator.get(arousal_seed, subject_seed, noise_seed_reuse)


def _main():
    #init_tf_gpus()
    datagenerator = tf.data.Dataset.from_generator(
        generator=_DataGenerator("../Logs/wgan-gp-big/model_gen"),
        output_types=(tf.float32, tf.float32)
    )

    for a,b in datagenerator.take(2):
        print(a)
        print(b)


if __name__ == '__main__':
    os.chdir("./..")
    _main()
