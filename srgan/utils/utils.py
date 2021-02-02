import tensorflow as tf
import matplotlib.pyplot as plt

def test_show_image(model,test_dataset,test_image_height,test_image_width,test_batch_size):
  for images in test_dataset.take(1):
    for i in range(test_batch_size):
      hr_image = images[0][i]
      lr_image = images[1][i]
      fig, ax = plt.subplots(1, 3, figsize=(40,40))
      ax[0].imshow(tf.cast((hr_image+1)*127.5, tf.uint8))
      ax[0].set_title("HR Image")
      ax[1].imshow(tf.cast(tf.image.resize((lr_image),size=(test_image_height,test_image_width)),tf.uint8))
      ax[1].set_title("LR Image")
      sr_image = model(tf.expand_dims(lr_image,0))
      sr_image = tf.squeeze(sr_image)
      ax[2].imshow(tf.cast((sr_image+1)*127.5,tf.uint8))
      ax[2].set_title("SR Image")
    plt.show()
