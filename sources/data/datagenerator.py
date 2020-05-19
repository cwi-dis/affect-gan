import tensorflow as tf
import os
#from main import init_tf_gpus
#from models.TAGAN import *


class _DataGenerator:
    def __init__(self, file_path, batch_size, noise_dim, class_conditioned, subject_conditioned, categorical_sampling):
        self.file_path = file_path
        self.batch_size = batch_size
        self.noise_dim = noise_dim
        self.class_conditioned = class_conditioned
        self.subject_conditioned = subject_conditioned
        self.categorical_sampling = categorical_sampling

        self.generator = tf.keras.models.load_model(
            filepath=self.file_path
        )

    def __call__(self, *args, **kwargs):
        while True:
            seed = tf.random.normal(shape=[self.batch_size, self.noise_dim])
            if self.categorical_sampling:
                generator_class_inputs = tf.one_hot(tf.random.uniform([self.batch_size], maxval=2, dtype=tf.int32), depth=2)
                generator_subject_inputs = tf.one_hot(tf.random.uniform([self.batch_size], maxval=29, dtype=tf.int32), depth=29)
            else:
                generator_class_inputs = tf.keras.activations.softmax(tf.random.normal([self.batch_size, 2], stddev=2))
                generator_subject_inputs = tf.keras.activations.softmax(tf.random.normal([self.batch_size, 29], stddev=4))
            if self.class_conditioned:
                seed = tf.concat([seed, generator_class_inputs], axis=-1)
            if self.subject_conditioned:
                seed = tf.concat([seed, generator_subject_inputs], axis=-1)
            gen_sig = self.generator(seed, training=False)
            yield gen_sig, generator_class_inputs, generator_subject_inputs


class DatasetGenerator:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __call__(self, generator_path, class_conditioned, subject_conditioned, categorical_sampling, no_subject_output=False, *args, **kwargs):
        noise_dim = 100 if subject_conditioned else 129
        datagenerator = tf.data.Dataset.from_generator(
            generator=_DataGenerator(generator_path, self.batch_size, noise_dim, class_conditioned, subject_conditioned, categorical_sampling),
            output_types=(tf.float32, tf.float32, tf.float32),
            output_shapes=((self.batch_size, 500, 1), (self.batch_size, 2), (self.batch_size, 29))
        )

        if no_subject_output:
            datagenerator = datagenerator.map(lambda signal, label, subject: (signal, label))
        return datagenerator


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
