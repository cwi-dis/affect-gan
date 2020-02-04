import tensorflow as tf
import tensorflow.keras.layers as layers


class ResNETBlock(layers.Layer):

    def __init__(self, filter_channels, kernel_sizes=None, expand_residual=True):
        super(ResNETBlock, self).__init__()

        if kernel_sizes is None or len(kernel_sizes) is not 3:
            kernel_sizes = [8, 5, 3]
        self.expand_residual = expand_residual

        self.conv_0 = layers.Conv1D(
            filters=filter_channels,
            kernel_size=kernel_sizes[0],
            strides=1,
            padding="same"
        )
        self.batchnorm_0 = layers.BatchNormalization()
        self.activation_0 = layers.Activation('relu')

        self.conv_1 = layers.Conv1D(
            filters=filter_channels,
            kernel_size=kernel_sizes[1],
            strides=1,
            padding="same"
        )
        self.batchnorm_1 = layers.BatchNormalization()
        self.activation_1 = layers.Activation('relu')

        self.conv_2 = layers.Conv1D(
            filters=filter_channels,
            kernel_size=kernel_sizes[2],
            strides=1,
            padding="same"
        )
        self.batchnorm_2 = layers.BatchNormalization()

        self.conv_residual = layers.Conv1D(
            filters=filter_channels,
            kernel_size=1,
            strides=1,
            padding="same"
        )
        self.batchresidual = layers.BatchNormalization()

        self.merge_residual = layers.Add()
        self.out_activation = layers.Activation('relu')

    def call(self, inputs, **kwargs):
        x = self.conv_0(inputs)
        x = self.batchnorm_0(x)
        x = self.activation_0(x)
        x = self.conv_1(x)
        x = self.batchnorm_1(x)
        x = self.activation_1(x)
        x = self.conv_2(x)
        x = self.batchnorm_2(x)

        if self.expand_residual:
            y = self.conv_residual(inputs)
            y = self.batchresidual(y)
        else:
            y = self.batchresidual(y)

        z = self.merge_residual([y, x])
        return self.out_activation(z)


class ResNET(tf.keras.Model):

    def __init__(self, num_classes):
        super(ResNET, self).__init__()

        self.resnet_block_1 = ResNETBlock(filter_channels=64)
        self.resnet_block_2 = ResNETBlock(filter_channels=128)

        self.global_avg_pool = layers.GlobalAveragePooling1D()
        self.dense_output = layers.Dense(units=num_classes, activation='sigmoid')

    def call(self, inputs, training=None, mask=None):
        x = self.input_norm(inputs)
        x_r1 = self.resnet_block_1(x)
        x_r2 = self.resnet_block_2(x_r1)
        out = self.global_avg_pool(x_r2)
        return self.dense_output(out)
