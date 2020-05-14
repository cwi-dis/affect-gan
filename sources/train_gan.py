import os
import tensorflow as tf
from util.CustomLosses import discriminator_loss, generator_loss, wgangp_critic_loss
from util.CustomMetrics import discriminator_accuracy
from data.inspector import plot_generated_signals, plot_to_image

from models.TAGAN import Generator, Discriminator

class GAN_Trainer():

    def __init__(
            self,
            mode,
            batch_size,
            hparams,
            logdir,
            num_classes,
            train_steps=200000,
            save_image_every_n_steps=1000,
            n_critic=1,
            noise_dim=125
    ):
        self.mode = mode
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.train_steps = train_steps
        self.noise_dim = noise_dim
        self.num_classes = num_classes
        self.conditional = num_classes != 0
        self.save_image_every_n_steps = save_image_every_n_steps

        self.discriminator = Discriminator(self.conditional, hparams)
        self.generator = Generator(n_signals=3)
        dis_lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, 50000, 0.666)
        gen_lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay(0.001, 50000/n_critic, 0.666)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr_decay, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=dis_lr_decay, beta_1=0.5, beta_2=0.9)
        self.train = self.train_vanilla if mode is "vanilla_gan" else self.train_wgangp

        self.discriminator_path = os.path.join(logdir, "model_dis")
        self.generator_path = os.path.join(logdir, "model_gen")
        self.summary_writer = tf.summary.create_file_writer(logdir=logdir)

    def train_wgangp(self, dataset):
        test_seed_0 = tf.random.normal([1, self.noise_dim])
        if self.conditional:
            test_seed_1 = tf.concat([test_seed_0, tf.ones(shape=(1, 1), dtype=tf.float32)], axis=-1)
            test_seed_0 = tf.concat([test_seed_0, tf.zeros(shape=(1, 1), dtype=tf.float32)], axis=-1)
        train_step = 1
        gen_loss, gen_classification_loss = 0, 0
        for batch, labels in dataset:
            critic_loss, classification_loss = self.train_step_wgangp_critic(batch, labels)

            if train_step % self.n_critic == 0:
                gen_loss, gen_classification_loss = self.train_step_wgangp_generator(batch)

            with self.summary_writer.as_default():
                tf.summary.scalar("critic_loss", critic_loss, step=train_step)
                tf.summary.scalar("generator_loss", gen_loss, step=train_step)
                tf.summary.scalar("classloss.critic", classification_loss, step=train_step)
                tf.summary.scalar("classloss.generator", gen_classification_loss, step=train_step)

            if train_step % self.save_image_every_n_steps == 0 or train_step == 1:
                tf.print("Current Train Step: %d, Critic Loss: %3f, Generator Loss: %3f" % (train_step, critic_loss, gen_loss))
                if self.conditional:
                    tf.print("BCE Loss: Critic: %3f, Generator: %3f" % (classification_loss, gen_classification_loss))
                gen_signals_0 = self.generator(test_seed_0, training=False)
                gen_signals_1 = None
                if self.conditional:
                    gen_signals_1 = self.generator(test_seed_1, training=False)
                fig = plot_generated_signals(gen_signals_0, gen_signals_1)
                img = plot_to_image(fig)
                with self.summary_writer.as_default():
                    tf.summary.image("Generated Signals", img, step=train_step)

            train_step += 1
            if train_step > self.train_steps:
                self.discriminator.save(self.discriminator_path)
                self.generator.save(self.generator_path)
                break

    def train_vanilla(self, dataset):
        test_seed = tf.random.normal([2, self.noise_dim])
        train_step = 1

        for batch in dataset:
            gen_loss, dis_loss, fake_acc, real_acc = self.train_step_vanilla(batch)

            with self.summary_writer.as_default():
                tf.summary.scalar("generator_loss", gen_loss, step=train_step)
                tf.summary.scalar("discriminator_loss", dis_loss, step=train_step)
                tf.summary.scalar("fake accuracy", fake_acc, step=train_step)
                tf.summary.scalar("real accuracy", real_acc, step=train_step)

            if train_step % self.save_image_every_n_steps == 0 or train_step == 1:
                tf.print("Current Train Step: %d, Generator Loss: %3f, Discriminator Loss: %3f, Fake Acc.: %3f, Real Acc.: %3f" % (train_step, gen_loss, dis_loss, fake_acc, real_acc))
                gen_signals = self.generator(test_seed, training=False)
                fig = plot_generated_signals(gen_signals, 1)
                img = plot_to_image(fig)
                with self.summary_writer.as_default():
                    tf.summary.image("Generated Signals", img, step=train_step)

            train_step += 1
            if train_step > self.train_steps:
                break

    @tf.function
    def train_step_wgangp_critic(self, real_sig, real_labels):
        generator_inputs = tf.random.normal([self.batch_size, self.noise_dim])
        generator_class_inputs = tf.cast(tf.random.uniform([self.batch_size, 1], maxval=self.num_classes, dtype=tf.int32), dtype=tf.float32)
        if self.conditional:
            generator_inputs = tf.concat([generator_inputs, generator_class_inputs], axis=-1)
        epsilon = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0, maxval=1)

        with tf.GradientTape(persistent=True) as tape:
            fake_sig = self.generator(generator_inputs, training=True)

            dif = tf.math.subtract(fake_sig, real_sig)
            interpolated_sig = tf.math.add(real_sig, tf.math.multiply(epsilon, dif))

            _, real_out, real_class_pred = self.discriminator(real_sig, training=True)
            _, fake_out, fake_class_pred = self.discriminator(fake_sig, training=True)

            _, interpolated_out, __ = self.discriminator(interpolated_sig, training=True)
            interpolated_gradients = tf.gradients(interpolated_out, [interpolated_sig])[0]

            critic_loss = wgangp_critic_loss(real_out, fake_out, interpolated_gradients)
            classification_loss_real = tf.reduce_mean(tf.metrics.binary_crossentropy(real_labels, real_class_pred))
            classification_loss_gen = tf.reduce_mean(tf.metrics.binary_crossentropy(generator_class_inputs, fake_class_pred))
            classification_loss =  (classification_loss_real-0.3)# + classification_loss_gen)
            if self.conditional:
                critic_loss += classification_loss

        critic_gradients = tape.gradient(critic_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(critic_gradients, self.discriminator.trainable_variables))
        del tape

        return critic_loss-classification_loss, classification_loss

    @tf.function
    def train_step_wgangp_generator(self, batch):
        generator_inputs = tf.random.normal([self.batch_size*2, self.noise_dim])
        generator_class_inputs = tf.cast(tf.random.uniform([self.batch_size*2, 1], maxval=self.num_classes, dtype=tf.int32), dtype=tf.float32)
        if self.conditional:
            generator_inputs = tf.concat([generator_inputs, generator_class_inputs], axis=-1)

        with tf.GradientTape() as gen_tape:
            fake_sig = self.generator(generator_inputs, training=True)
            _, fake_critic_out, class_pred = self.discriminator(fake_sig, training=True)
            fake_critic_loss = -tf.reduce_mean(fake_critic_out)
            classification_loss = 0.5*tf.reduce_mean(tf.metrics.binary_crossentropy(generator_class_inputs, class_pred))
            if self.conditional:
                fake_critic_loss += classification_loss

        generator_gradients = gen_tape.gradient(fake_critic_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        del gen_tape

        return fake_critic_loss, classification_loss

    @tf.function
    def train_step_vanilla(self, batch):
        generator_inputs = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
        #with tf.GradientTape(persistent=True) as tape:
            fake_sig = self.generator(generator_inputs, training=True)

            real_out, _ = self.discriminator(batch, training=True)
            fake_out, _ = self.discriminator(fake_sig, training=True)

            gen_loss = generator_loss(fake_out)
            dis_loss = discriminator_loss(real_out, fake_out)
            fake_acc, real_acc = discriminator_accuracy(fake_out, real_out)

        fake_acc, real_acc = discriminator_accuracy(fake_out, real_out)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        dis_gradients = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(dis_gradients, self.discriminator.trainable_variables))

        return gen_loss, dis_loss, fake_acc, real_acc
