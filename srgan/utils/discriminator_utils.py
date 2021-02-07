import tensorflow as tf

class Discriminator_Block(tf.keras.layers.Layer):
  def __init__(self,filters,bn,**kwargs):
    super(Discriminator_Block,self).__init__()
    self.filters = filters
    self.bn = bn
    self.conv_layer_1 = tf.keras.layers.Conv2D(filters,3,1,'same')
    self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.8)
    self.lrelu_layer = tf.keras.layers.LeakyReLU(alpha=0.2)

    self.conv_layer_2 = tf.keras.layers.Conv2D(filters,2,1,'same')

  def call(self,input_tensor,training=None):
    x = self.conv_layer_1(input_tensor)
    x = self.bn_layer(x,training=training)
    x = self.lrelu_layer(x)
    x = self.conv_layer_2(x)
    x = self.bn_layer(x,training=training)
    x = self.lrelu_layer(x)
    return x
