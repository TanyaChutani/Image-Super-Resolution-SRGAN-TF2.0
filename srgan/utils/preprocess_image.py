import tensorflow as tf

def train_preprocess_image(path,train_image_width,train_image_height,downsample_factor):
  image = tf.io.read_file(path)
  image = tf.image.decode_png(image, channels=3)
  hr_image = tf.image.random_crop(image,size=(train_image_width,train_image_height,3))
  lr_image = tf.image.resize(hr_image,size=(train_image_width//downsample_factor,train_image_height//downsample_factor),
                             method=tf.image.ResizeMethod.BILINEAR)
  hr_image = tf.cast(hr_image,tf.float32)/127.5 - 1
  lr_image = tf.cast(lr_image,tf.float32)

  return hr_image, lr_image

def test_preprocess_image(path,test_image_height,test_image_width,downsample_factor):
  image = tf.io.read_file(path)
  image = tf.image.decode_png(image, channels=3)
  hr_image = tf.image.resize(image,(test_image_height,test_image_width))
  lr_image = tf.image.resize(hr_image,size=(test_image_height//downsample_factor,test_image_width//downsample_factor),
                             method=tf.image.ResizeMethod.BILINEAR)
  hr_image = tf.cast(hr_image,tf.float32)/127.5 - 1
  lr_image = tf.cast(lr_image,tf.float32)

  return hr_image, lr_image
