import tensorflow as tf
from srgan.utils.discriminator_utils import Discriminator_Block

class Discriminator(tf.keras.layers.Layer):
    def __init__(self, filters=64, **kwargs):
        super(Discriminator, self).__init__()
        self.filters = filters
        self.make_discriminator_blocks = self.make_discriminator_block()
        self.global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        self.dense_layer = tf.keras.layers.Dense(1024)
        self.lrelu_layer = tf.keras.layers.LeakyReLU(alpha=0.2)
        self.model_output = tf.keras.layers.Dense(1, activation='sigmoid')

    def make_discriminator_block(self):
        label = []
        label.append(Discriminator_Block(self.filters, bn=False))
        for _ in range(3):
            self.filters = self.filters * 2
            label.append(Discriminator_Block(filters=self.filters, bn=True))
        return tf.keras.Sequential(label)

    def call(self, input_tensor, training=None):
        x = self.make_discriminator_blocks(input_tensor, training=training)
        x = self.global_average_layer(x)
        x = self.dense_layer(x)
        x = self.lrelu_layer(x)
        x = self.model_output(x)
        return x
