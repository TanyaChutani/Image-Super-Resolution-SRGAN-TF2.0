import tensorflow as tf
from srgan.models.generator import Generator

class Generator_MSE_Model(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Generator_MSE_Model, self).__init__()
        self.generator_model = Generator()

    def compile(self, generator_optimizer, generator_loss):
        super(Generator_MSE_Model, self).compile()
        self.generator_optimizer = generator_optimizer
        self.generator_loss = generator_loss

    def train_step(self, data):
        hr_image, lr_image = data
        with tf.GradientTape() as t:
            sr_image = self.generator_model(lr_image, training=True)
            mse_loss = self.generator_loss(hr_image, sr_image)
        generator_gradients = t.gradient(mse_loss, self.trainable_variables)
        self.generator_optimizer.apply_gradients(zip(generator_gradients,
                                                     self.trainable_variables))
        return {"mse_loss": mse_loss}

    def call(self, lr_image):
        sr_image = self.generator_model(lr_image, training=False)
        return sr_image
