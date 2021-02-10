import tensorflow as tf

class SRGAN(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(SRGAN, self).__init__()
        self.generator_model = generator_model
        self.discriminator_model = discriminator_model
        self.content_loss = Content_Loss()
        self.mse = tf.keras.losses.MeanSquaredError(reduction='none')

    def compile(self, generator_optimizer, discriminator_optimizer, loss_fn, srgan_metrics):
        super(SRGAN, self).compile()
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.loss_fn = loss_fn
        self.srgan_metrics = srgan_metrics

    def train_step(self, data):
        hr_image, lr_image = data
        batch_size = tf.shape(lr_image)[0]
        ones = tf.ones([batch_size])
        zeros = tf.zeros([batch_size])

        with tf.GradientTape() as discriminator_tape, tf.GradientTape() as generator_tape:
            sr_image = self.generator_model(lr_image, training=True)
            hr_output = self.discriminator_model(hr_image, training=True)
            sr_output = self.discriminator_model(sr_image, training=True)

            discriminator_loss_fake = tf.reduce_mean(self.loss_fn(zeros, sr_output))
            discriminator_loss_real = tf.reduce_mean(self.loss_fn(ones, hr_output))
            discriminator_loss = discriminator_loss_fake + discriminator_loss_real

            content_loss = self.content_loss(hr_image, sr_image)
            generator_loss = tf.reduce_mean(self.loss_fn(ones, sr_output))

            perceptual_loss = content_loss + 1e-3 * generator_loss
            psnr_metric = self.srgan_metrics(hr_image, sr_image)

        discriminator_gradients = discriminator_tape.gradient(discriminator_loss,
                                                              self.discriminator_model.trainable_variables)
        generator_gradients = generator_tape.gradient(perceptual_loss, self.generator_model.trainable_variables)

        self.discriminator_optimizer.apply_gradients(
            zip(discriminator_gradients, self.discriminator_model.trainable_variables))
        self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator_model.trainable_variables))

        return {"generator_loss": perceptual_loss,
                "discriminator_loss": discriminator_loss,
                "psnr_metrics": psnr_metric}

    def call(self, lr_image):
        sr_image = self.generator_model(lr_image, training=False)
        return sr_image
