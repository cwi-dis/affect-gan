import config

import tensorflow as tf
import tensorflow.keras.layers as layers


class ChannelDownResBlock(layers.Layer):
    def __init__(self, channels, kernel_size, w_norm_clip, activation=None):
        super(ChannelDownResBlock, self).__init__()
        self.out_channels = channels
        self.conv1 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same", activation=layers.LeakyReLU(alpha=0.2),
                                   kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1]))
        self.conv2 = layers.Conv1D(filters=channels, kernel_size=kernel_size, padding="same",
                                   kernel_constraint=tf.keras.constraints.MaxNorm(max_value=w_norm_clip, axis=[0,1]))

        self.pool = layers.AveragePooling1D(pool_size=2, strides=2)

        self.shortcut_conv = layers.Conv1D(filters=channels, kernel_size=1, padding="same")
        self.shortcut_pool = layers.AveragePooling1D(pool_size=2, strides=2)

        self.out_activation = activation

    def call(self, inputs, **kwargs):
        input_channels = inputs.shape.as_list()[-1]

        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.pool(x)

        if self.out_channels != input_channels:
            inputs = self.shortcut_conv(inputs)
        inputs = self.shortcut_pool(inputs)

        out = x + inputs

        if self.out_activation is not None:
            out = self.out_activation(out)

        return out

class ChannelDownResLayer(layers.Layer):
    def __init__(self, channels_out, dropout_rate=0.4, kernel_size=7, w_norm_clip=2, last_layer=False, **kwargs):
        super(ChannelDownResLayer, self).__init__(**kwargs)
        self.last_layer = last_layer

        self.down_resblock = ChannelDownResBlock(channels_out, kernel_size, w_norm_clip)
        self.lrelu = layers.LeakyReLU()
        self.dropout = layers.Dropout(rate=dropout_rate)

    def call(self, inputs, **kwargs):
        x = self.down_resblock(inputs)

        if not self.last_layer:
            x = self.dropout(self.lrelu(x), training=kwargs["training"])

        return x


class DeepCNN(tf.keras.Model):

    def __init__(self, hparams, *args, **kwargs):
        super(DeepCNN, self).__init__(*args, **kwargs)
        self.layers_count = hparams[config.HP_DEEP_LAYERS]
        self.dual_output = hparams[config.HP_LOSS_TYPE] == "DUAL_BCE"
        self.input_len = 500

        self.down_res_layers = [ChannelDownResLayer
            (
                hparams[config.HP_DEEP_CHANNELS] * ((l+3) // 2),
                kernel_size=max(3, hparams[config.HP_DEEP_KERNEL_SIZE] - 2*l)
            ) for l in range(self.layers_count - 1)]
        self.down_res_layer_final = ChannelDownResLayer(
                hparams[config.HP_DEEP_CHANNELS] * ((self.layers_count+2) // 2),
                kernel_size=max(3, hparams[config.HP_DEEP_KERNEL_SIZE] - 2*(self.layers_count-1)),
                last_layer=True)

        self.feature_pool = layers.GlobalAveragePooling1D()
        self.lrelu_out = layers.LeakyReLU()

        if hparams[config.HP_LOSS_TYPE] == "MSE":
            activation = None
        else:
            activation = 'sigmoid'

        self.dense_out = layers.Dense(units=2, activation=activation)

        self.dense_out_a = layers.Dense(units=1, activation=activation, name="arousal_class")
        self.dense_out_v = layers.Dense(units=1, activation=activation, name="valence_class")

    def call(self, inputs, training=None, mask=None):
        x = inputs

        for i in range(self.layers_count - 1):
            x = self.down_res_layers[i](x, training=training)

        x = self.down_res_layer_final(x, training=training)

        x = self.feature_pool(x)
        x = self.lrelu_out(x)

        if self.dual_output:
            return self.dense_out_a(x), self.dense_out_v(x)
        else:
            return self.dense_out_a(x)

    def model(self):
        x = layers.Input(shape=(500, 5))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))
