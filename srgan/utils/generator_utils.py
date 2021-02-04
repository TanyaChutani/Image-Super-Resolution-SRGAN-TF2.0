import tensorflow as tf

class Residual_Block(tf.keras.layers.Layer):
  def __init__(self,**kwargs):
    super(Residual_Block,self).__init__(**kwargs)
    self.conv_layer = tf.keras.layers.Conv2D(64,3,1,padding='same')
    self.bn_layer = tf.keras.layers.BatchNormalization(momentum=0.8)
    self.prelu_layer = tf.keras.layers.PReLU(shared_axes=[1,2])

  def call(self,input_tensor,training=None):
    x = self.conv_layer(input_tensor)
    x = self.bn_layer(x,training=training)
    x = self.prelu_layer(x)
    x = self.conv_layer(x)
    x = self.bn_layer(x,training=training)
    x = tf.add(input_tensor,x)
    return x

class Upsample_Block(tf.keras.layers.Layer):
  def __init__(self,**kwargs):
    super(Upsample_Block,self).__init__(**kwargs)
    self.conv_layer = tf.keras.layers.Conv2D(256,3,1,padding='same')
    self.subpixel_layer = self.sub_pixel_conv2D(2)
    self.prelu_layer = tf.keras.layers.PReLU(shared_axes=[1,2])

  @staticmethod
  def sub_pixel_conv2D(scale):
    return tf.keras.layers.Lambda(lambda x:tf.nn.depth_to_space(x,scale))

  def call(self,input_tensor):
    x = self.conv_layer(input_tensor)
    x = self.subpixel_layer(x)
    x = self.prelu_layer(x)
    return x
