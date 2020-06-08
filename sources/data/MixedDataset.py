import tensorflow as tf
import numpy as np
import os
from data.datagenerator import DatasetGenerator
from data.dataloader import Dataloader
#from main import init_tf_gpus

class MixedDataset:
    def __init__(self, path, batch_size, features, subject_conditioned, class_categorical_sampling, subject_categorical_sampling,
                 discriminator_class_conditioned, argmaxed_label=False):
        self.batch_chunks = 16
        self.batch_size = batch_size // self.batch_chunks

        self.fake_datagenerator = DatasetGenerator(batch_size=self.batch_chunks,
                                             path=path,
                                             subject_conditioned=subject_conditioned,
                                             class_categorical_sampling=class_categorical_sampling,
                                             subject_categorical_sampling=subject_categorical_sampling,
                                             discriminator_class_conditioned=discriminator_class_conditioned,
                                             no_subject_output=True,
                                             argmaxed_label=argmaxed_label
                                             )

        self.real_dataloader = Dataloader("5000d", features, normalized=True, continuous_labels=False)

    def __call__(self, out_subject, *args, **kwargs):
        fake_dataset = self.fake_datagenerator()
        real_dataset = self.real_dataloader(mode="train", batch_size=self.batch_chunks, leave_out=out_subject, one_hot=True)

        mixed_dataset = tf.data.experimental.sample_from_datasets(
            datasets=[fake_dataset, real_dataset]
        )

        mixed_dataset = mixed_dataset.batch(batch_size=self.batch_size)
        mixed_dataset = mixed_dataset.map(lambda data, label:
                                          (tf.reshape(data, (-1, 500, 2)), tf.reshape(label, (-1, 2))))
        mixed_dataset = mixed_dataset.prefetch(buffer_size=2)

        return mixed_dataset

def _main():
    #init_tf_gpus()
    datagenerator = MixedDataset(
        path="../Logs/loso-wgan-gp20200603-122650/subject-4-out",
        batch_size=4,
        features=["ecg", "gsr"],
        subject_conditioned=True,
        class_categorical_sampling=False,
        subject_categorical_sampling=False,
        discriminator_class_conditioned=False,
        argmaxed_label=False
    )

    datag = datagenerator(out_subject=4)

    for d in datag.take(1):
        print(d)



if __name__ == '__main__':
    os.chdir("./..")
    _main()