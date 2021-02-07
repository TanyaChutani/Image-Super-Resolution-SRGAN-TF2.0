import tensorflow as tf

class Content_Loss(tf.keras.losses.Loss):
    def __init__(self):
        super(Content_Loss, self).__init__()
        self.VGG_Model = self.vgg_model()
        self.mse_loss = tf.keras.losses.MeanSquaredError()

    @staticmethod
    def vgg_model():
        vgg = tf.keras.applications.VGG19(include_top=False,
                                          weights='imagenet',
                                          input_shape=(None, None, 3))
        vgg.trainable = False
        vgg_model = tf.keras.Model(vgg.input, vgg.get_layer('block5_conv4').output)
        return vgg_model

    def content_loss(self, hr_image, sr_image):
        hr_image = tf.keras.applications.vgg19.preprocess_input((hr_image + 1) * 127.5)
        sr_image = tf.keras.applications.vgg19.preprocess_input((sr_image + 1) * 127.5)

        hr_output = self.VGG_Model(hr_image) / 12.75
        sr_output = self.VGG_Model(sr_image) / 12.75

        mse_loss = tf.math.square(hr_output - sr_output)
        mse_loss = tf.math.reduce_mean(mse_loss, axis=-1)
        mse_loss = tf.reduce_sum(mse_loss)

        return mse_loss

    @tf.function
    def call(self, hr_image, sr_image):
        content_loss = self.content_loss(hr_image, sr_image)
        return content_loss
