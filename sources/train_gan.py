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
            n_signals=1,
            leave_out=1,
            class_conditional=True,
            subject_conditional=True,
            train_steps=200000,
            save_image_every_n_steps=1000,
            n_critic=1
    ):
        self.mode = mode
        self.batch_size = batch_size
        self.n_critic = n_critic
        self.train_steps = train_steps
        self.noise_dim = 100 if subject_conditional else 129
        self.num_classes = num_classes
        self.num_subjects = 29
        self.leave_out = leave_out
        self.class_conditional = class_conditional
        self.subject_conditional = subject_conditional
        self.save_image_every_n_steps = save_image_every_n_steps
        self.n_signals = n_signals

        self.discriminator = Discriminator(self.class_conditional, self.subject_conditional, hparams)
        self.generator = Generator(n_signals=n_signals)
        self.classification_loss_factor = 0.25 if (self.class_conditional and self.subject_conditional) else 0.5
        dis_lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay(0.0008, 50000, 0.8)
        gen_lr_decay = tf.keras.optimizers.schedules.InverseTimeDecay(0.002, 50000/n_critic, 0.8)
        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=gen_lr_decay, beta_1=0.5, beta_2=0.99)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=dis_lr_decay, beta_1=0.5, beta_2=0.99)
        self.train = self.train_vanilla if mode is "vanilla_gan" else self.train_wgangp

        self.discriminator_path = os.path.join(logdir, "model_dis")
        self.generator_path = os.path.join(logdir, "model_gen")
        self.summary_writer = tf.summary.create_file_writer(logdir=logdir)
        self.da = 0.75
        self.ds = 0.025
        self.ga = 0.8
        self.gs = 0.3

    def train_wgangp(self, dataset):
        test_seed_0 = tf.random.normal([5, self.noise_dim])
        test_seed_1 = test_seed_0
        if self.class_conditional:
            test_seed_1 = tf.concat([test_seed_1, tf.tile([[0., 1.]], multiples=[5, 1])], axis=-1)
            test_seed_0 = tf.concat([test_seed_0, tf.tile([[1., 0.]], multiples=[5, 1])], axis=-1)
        if self.subject_conditional:
            subject_seed = tf.one_hot(tf.random.uniform([5], maxval=self.num_subjects, dtype=tf.int32),depth=self.num_subjects)
            test_seed_1 = tf.concat([test_seed_1, subject_seed], axis=-1)
            test_seed_0 = tf.concat([test_seed_0, subject_seed], axis=-1)

        train_step = 1
        for batch, labels, subject in dataset:
            labels = tf.squeeze(tf.one_hot(tf.cast(labels, tf.int32), depth=self.num_classes))
            subject = tf.squeeze(tf.one_hot(tf.cast(subject, tf.int32), depth=self.num_subjects))
            critic_loss, classification_loss, subject_loss = self.train_step_wgangp_critic(batch, labels, subject)

            if train_step % self.n_critic == 0:
                gen_loss, gen_classification_loss, gen_subject_loss = self.train_step_wgangp_generator()

            if train_step % self.save_image_every_n_steps == 0:
                with self.summary_writer.as_default():
                    tf.summary.scalar("critic_loss", critic_loss, step=train_step)
                    tf.summary.scalar("generator_loss", gen_loss, step=train_step)
                    tf.summary.scalar("classloss.critic", classification_loss, step=train_step)
                    tf.summary.scalar("classloss.generator", gen_classification_loss, step=train_step)
                    tf.summary.scalar("subjectloss.critic", subject_loss, step=train_step)
                    tf.summary.scalar("subjectloss.generator", gen_subject_loss, step=train_step)
                    tf.summary.scalar("Gen. attention Gamma", self.generator.get_layer("att0").gamma, step=train_step)
                    tf.summary.scalar("Critic attention Gamma", self.discriminator.get_layer("att0").gamma, step=train_step)

                tf.print("Current Train Step: %d, Critic Loss: %3f, Generator Loss: %3f" % (train_step, critic_loss, gen_loss))
                if self.class_conditional:
                    tf.print("Class Loss: Critic: %3f, Generator: %3f" % (classification_loss, gen_classification_loss))
                if self.subject_conditional:
                    tf.print("Subject Loss: Critic: %3f, Generator: %3f" % (subject_loss, gen_subject_loss))
                gen_signals_0 = self.generator(test_seed_0, training=False)
                gen_signals_1 = self.generator(test_seed_1, training=False)
                fig = plot_generated_signals(gen_signals_0, gen_signals_1)
                img = plot_to_image(fig)
                with self.summary_writer.as_default():
                    tf.summary.image("Generated Signals", img, step=train_step)

            if train_step % 50000 == 0:
                self.discriminator.save(os.path.join(self.discriminator_path, "%d"%train_step))
                self.generator.save(os.path.join(self.generator_path, "%d"%train_step))

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
    def train_step_wgangp_critic(self, real_sig, real_labels, subject):
        generator_inputs = tf.random.normal([self.batch_size, self.noise_dim])
        generator_class_inputs = tf.one_hot(tf.random.uniform([self.batch_size], maxval=self.num_classes, dtype=tf.int32), depth=self.num_classes)
        generator_subject_inputs = tf.one_hot(tf.random.uniform([self.batch_size], maxval=self.num_subjects, dtype=tf.int32), depth=self.num_subjects)
        #generator_class_inputs = real_labels
        #generator_subject_inputs = subject
        if self.class_conditional:
            generator_inputs = tf.concat([generator_inputs, generator_class_inputs], axis=-1)
        if self.subject_conditional:
            generator_inputs = tf.concat([generator_inputs, generator_subject_inputs], axis=-1)

        epsilon = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0, maxval=1)

        with tf.GradientTape(persistent=True) as tape:
            fake_sig = self.generator(generator_inputs, training=True)

            dif = tf.math.subtract(fake_sig, real_sig)
            interpolated_sig = tf.math.add(real_sig, tf.math.multiply(epsilon, dif))

            _, real_out, real_class_pred, real_subject_pred = self.discriminator(real_sig, training=True)
            _, fake_out, fake_class_pred, fake_subject_pred = self.discriminator(fake_sig, training=True)

            _, interpolated_out, __, ___ = self.discriminator(interpolated_sig, training=True)
            interpolated_gradients = tf.gradients(interpolated_out, [interpolated_sig])[0]

            critic_loss = wgangp_critic_loss(real_out, fake_out, interpolated_gradients)

            classification_loss_real = 0 
            subject_loss_real = 0
            if self.class_conditional:
                classification_loss_real = tf.reduce_mean(tf.keras.losses.kld(real_labels, real_class_pred))
                critic_loss += self.da * classification_loss_real

            if self.subject_conditional:
                subject_loss_real = tf.reduce_mean(tf.keras.losses.kld(subject, real_subject_pred))
                critic_loss += self.ds * subject_loss_real

        critic_gradients = tape.gradient(critic_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(critic_gradients, self.discriminator.trainable_variables))
        del tape

        return critic_loss-classification_loss_real-subject_loss_real, tf.cast(classification_loss_real, tf.float32),tf.cast(subject_loss_real, tf.float32)

    @tf.function
    def train_step_wgangp_generator(self):
        generator_inputs = tf.random.normal([self.batch_size*2, self.noise_dim])
        generator_class_inputs = tf.one_hot(tf.random.uniform([self.batch_size*2], maxval=self.num_classes, dtype=tf.int32), depth=self.num_classes)
        generator_subject_inputs = tf.one_hot(tf.random.uniform([self.batch_size*2], maxval=self.num_subjects, dtype=tf.int32), depth=self.num_subjects)
        if self.class_conditional:
            generator_inputs = tf.concat([generator_inputs, generator_class_inputs], axis=-1)
        if self.subject_conditional:
            generator_inputs = tf.concat([generator_inputs, generator_subject_inputs], axis=-1)

        with tf.GradientTape() as gen_tape:
            fake_sig = self.generator(generator_inputs, training=True)
            _, fake_critic_out, class_pred, subject_pred = self.discriminator(fake_sig, training=True)
            fake_critic_loss = -tf.reduce_mean(fake_critic_out)

            classification_loss = 0
            subject_loss = 0

            if self.class_conditional:
                classification_loss = tf.reduce_mean(tf.keras.losses.kld(generator_class_inputs, class_pred))
                fake_critic_loss += self.ga * classification_loss

            if self.subject_conditional:
                subject_loss = tf.reduce_mean(tf.keras.losses.kld(generator_subject_inputs, subject_pred))
                fake_critic_loss += self.gs * subject_loss

        generator_gradients = gen_tape.gradient(fake_critic_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        del gen_tape

        return fake_critic_loss-classification_loss-subject_loss, tf.cast(classification_loss, tf.float32), tf.cast(subject_loss, tf.float32)

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
