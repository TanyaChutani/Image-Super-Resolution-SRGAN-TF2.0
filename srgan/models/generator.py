import tensorflow as tf

class Generator(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Generator, self).__init__(name='generator', **kwargs)
        self.conv_layer_1 = tf.keras.layers.Conv2D(64, 9, 1, padding='same')
        self.prelu_layer = tf.keras.layers.PReLU(shared_axes=[1, 2])
        self.make_residual_block = self.make_residual_blocks()
        self.conv_layer_2 = tf.keras.layers.Conv2D(64, 3, 1, padding='same')
        self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.8)
        self.make_upsample_block = self.make_upsample_blocks()
        self.conv_layer_3 = tf.keras.layers.Conv2D(3, 9, 1, padding='same', activation='tanh')

    def make_residual_blocks(self):
        label = []
        for _ in range(16):
            label.append(Residual_Block())
        return tf.keras.Sequential(label)

    def make_upsample_blocks(self):
        label = []
        for _ in range(2):
            label.append(Upsample_Block())
        return tf.keras.Sequential(label)

    def call(self, input_tensor, training=None):
        block_1 = self.conv_layer_1(input_tensor)
        block_1 = self.prelu_layer(block_1)
        residual_block = block_1
        residual_block = self.make_residual_block(residual_block, training=training)
        block_2 = self.conv_layer_2(residual_block)
        block_2 = self.bn_layer(block_2, training=training)
        block_3 = tf.add(block_1, block_2)
        block_3 = self.make_upsample_block(block_3)
        return self.conv_layer_3(block_3)
