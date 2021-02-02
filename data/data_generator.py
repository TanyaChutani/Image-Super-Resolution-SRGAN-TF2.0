import tensorflow as tf
from srgan.utils.preprocess_image import train_preprocess_image, test_preprocess_image

def train_data_generator(path,train_batch_size):
  dataset = tf.data.Dataset.list_files(path,tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(train_preprocess_image,tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(train_batch_size,drop_remainder=True)
  dataset.repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset

def test_data_generator(path,test_batch_size):
  dataset = tf.data.Dataset.list_files(path,tf.data.experimental.AUTOTUNE)
  dataset = dataset.map(test_preprocess_image,tf.data.experimental.AUTOTUNE)
  dataset = dataset.batch(test_batch_size,drop_remainder=True)
  dataset.repeat()
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
  return dataset