import tensorflow as tf
import os
#from main import init_tf_gpus
from models.TAGAN import *


class _DataGenerator:
    def __init__(self, file_path):
        self.file_path = file_path
        self.generator = tf.keras.models.load_model(
            filepath="../Logs/wgan-gp-big/model_gen"
        )

    def __call__(self, *args, **kwargs):
        while True:
            seed = tf.random.normal(shape=[1, 125])
            generator_class_inputs = tf.cast(
                tf.random.uniform([1, 1], maxval=2, dtype=tf.int32), dtype=tf.float32)
            generator_inputs = tf.concat([seed, generator_class_inputs], axis=-1)
            gen_sig = self.generator(generator_inputs, training=False)
            yield gen_sig, generator_class_inputs


class DatasetGenerator:
    def __init__(self, generator_path, batch_size):
        self.generator_path = generator_path
        self.batch_size = batch_size

    def __call__(self, *args, **kwargs):
        datagenerator = tf.data.Dataset.from_generator(
            generator=_DataGenerator(self.generator_path),
            output_types=(tf.float32, tf.float32)
        )
        datagenerator = datagenerator.batch(self.batch_size)

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