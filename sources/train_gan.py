import tensorflow as tf
from util.CustomLosses import discriminator_loss, generator_loss, wgangp_critic_loss
from util.CustomMetrics import discriminator_accuracy
from data.inspector import plot_generated_signals, plot_to_image

class GAN_Trainer():

    def __init__(
            self,
            mode,
            batch_size,
            generator,
            discriminator,
            logdir,
            n_epochs=10,
            iter_per_epoch=3000,
            save_image_every_n_steps=250,
            n_critic=1,
            noise_dim=125
    ):
        self.mode = mode
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.n_critic = n_critic
        self.iterations_per_epoch = iter_per_epoch
        self.noise_dim = noise_dim
        self.generator = generator
        self.discriminator = discriminator
        self.save_image_every_n_steps = save_image_every_n_steps

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001, beta_1=0.5, beta_2=0.9)
        self.train = self.train_vanilla if mode is "vanilla_gan" else self.train_wgangp

        self.summary_writer = tf.summary.create_file_writer(logdir=logdir)

    def train_wgangp(self, dataset):
        test_seed = tf.random.normal([2, self.noise_dim])
        train_step = 1
        gen_loss = 0
        for epoch in range(self.n_epochs):
            for batch in dataset:
                critic_loss = self.train_step_wgangp_critic(batch)

                if train_step % self.n_critic == 0:
                    gen_loss = self.train_step_wgangp_generator(batch)

                with self.summary_writer.as_default():
                    tf.summary.scalar("critic_loss", critic_loss, step=train_step)
                    tf.summary.scalar("generator_loss", gen_loss, step=train_step)

                if train_step % self.save_image_every_n_steps == 0 or train_step == 1:
                    tf.print("Current Train Step: %d, Critic Loss: %3f, Generator Loss: %3f" % (train_step, critic_loss, gen_loss))
                    gen_signals = self.generator(test_seed, training=False)
                    fig = plot_generated_signals(gen_signals, 1)
                    img = plot_to_image(fig)
                    with self.summary_writer.as_default():
                        tf.summary.image("Generated Signals", img, step=train_step)

                train_step += 1
            tf.print("End of epoch %d: train step %d" % (epoch, train_step))

    def train_vanilla(self, dataset):
        test_seed = tf.random.normal([2, self.noise_dim])
        train_step = 1

        for epoch in range(self.n_epochs):
            for batch in dataset:
                gen_loss, dis_loss, fake_acc, real_acc = self.train_step_vanilla(batch, train_step)

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
            tf.print("End of epoch %d: train step %d" % (epoch, train_step))

    @tf.function
    def train_step_wgangp_critic(self, real_sig):
        generator_inputs = tf.random.normal([self.batch_size, self.noise_dim])
        epsilon = tf.random.uniform(shape=[self.batch_size, 1, 1], minval=0, maxval=1)

        with tf.GradientTape(persistent=True) as tape:
            fake_sig = self.generator(generator_inputs, training=True)

            dif = tf.math.subtract(fake_sig, real_sig)
            interpolated_sig = tf.math.add(real_sig, tf.math.multiply(epsilon, dif))

            real_out = self.discriminator(real_sig, training=True)
            fake_out = self.discriminator(fake_sig, training=True)
            interpolated_out = self.discriminator(interpolated_sig, training=True)
            interpolated_gradients = tape.gradient(interpolated_out, interpolated_sig)

            critic_loss = wgangp_critic_loss(real_out, fake_out, interpolated_gradients)

        critic_gradients = tape.gradient(critic_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(zip(critic_gradients, self.discriminator.trainable_variables))
        del tape

        return critic_loss

    @tf.function
    def train_step_wgangp_generator(self, batch):
        generator_inputs = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape:
            fake_sig = self.generator(generator_inputs, training=True)
            fake_critic_out = self.discriminator(fake_sig, training=True)
            fake_critic_loss = -tf.reduce_mean(fake_critic_out)

        generator_gradients = gen_tape.gradient(fake_critic_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
        del gen_tape

        return fake_critic_loss

    @tf.function
    def train_step_vanilla(self, batch, train_step):
        generator_inputs = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            fake_sig = self.generator(generator_inputs, training=True)

            real_out = self.discriminator(batch, training=True)
            fake_out = self.discriminator(fake_sig, training=True)

            gen_loss = generator_loss(fake_out)
            dis_loss = discriminator_loss(real_out, fake_out)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        dis_gradients = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(dis_gradients, self.discriminator.trainable_variables))

        fake_acc, real_acc = discriminator_accuracy(fake_out, real_out)

        return gen_loss, dis_loss, fake_acc, real_acc
