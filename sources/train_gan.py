import tensorflow as tf
from util.CustomLosses import discriminator_loss, generator_loss
from data.inspector import plot_generated_signals, plot_to_image

class GAN_Trainer():
    def __init__(
            self,
            batch_size,
            n_epochs,
            iter_per_epoch,
            noise_dim,
            generator,
            discriminator,
            generator_lr,
            discriminator_lr,
            save_image_every_n_steps,
            logdir
    ):
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.iterations_per_epoch = iter_per_epoch
        self.noise_dim = noise_dim
        self.generator = generator
        self.discriminator = discriminator
        self.save_image_every_n_steps = save_image_every_n_steps

        self.generator_optimizer = tf.keras.optimizers.Adam(learning_rate=generator_lr, beta_1=0.5)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=discriminator_lr, beta_1=0.5)

        self.summary_writer = tf.summary.create_file_writer(logdir=logdir)

    def train(self, dataset):
        test_seed = tf.random.normal([2, self.noise_dim])
        train_step = 0

        for epoch in range(self.n_epochs):
            for batch in dataset.take(self.iterations_per_epoch):
                gen_loss, dis_loss = self.train_step(batch)

                with self.summary_writer.as_default():
                    tf.summary.scalar("generator_loss", gen_loss, step=train_step)
                    tf.summary.scalar("discriminator_loss", dis_loss, step=train_step)

                if train_step % self.save_image_every_n_steps == 0:
                    tf.print("Current Train Step: %d, Generator Loss: %3f, Discriminator Loss: %3f" % (train_step, gen_loss, dis_loss))
                    gen_signals = self.generator(test_seed, training=False)
                    fig = plot_generated_signals(gen_signals, 5)
                    img = plot_to_image(fig)
                    with self.summary_writer.as_default():
                        tf.summary.image("Generated Signals", img, step=train_step)

                train_step += 1



    @tf.function
    def train_step(self, batch):
        generator_inputs = tf.random.normal([self.batch_size, self.noise_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as dis_tape:
            fake_sig = self.generator(generator_inputs, training=True)

            real_out = self.discriminator(batch, training=True)
            fake_out = self.discriminator(fake_sig, training=True)

            gen_loss = generator_loss(fake_out)
            dis_loss = discriminator_loss(real_out, fake_out)

        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        dis_gradients = dis_tape.gradient(dis_loss, self.discriminator.trainable_variables)

        self.generator_optimizer.apply_gradients(zip(gen_gradients,self.generator.trainable_variables))
        self.discriminator_optimizer.apply_gradients(zip(dis_gradients, self.discriminator.trainable_variables))

        return gen_loss, dis_loss
