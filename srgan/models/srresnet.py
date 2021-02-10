import tensorflow as tf

class Generator_MSE_Model(tf.keras.models.Model):
    def __init__(self, **kwargs):
        super(Generator_MSE_Model, self).__init__()
        self.generator_model = generator_model

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


generator_learning_rate = 0.0001
generator_mse_model = Generator_MSE_Model(generator_model=generator_model)
generator_mse_model.generator_model.build((1, None, None, 3))
generator_mse_model.load_weights('/content/drive/MyDrive/srresnet_weights/')
callbacks = [tf.keras.callbacks.ModelCheckpoint('/content/drive/MyDrive/srresnet_weights/',
                                                monitor='mse_loss',
                                                save_best_only=False,
                                                save_weights_only=True,
                                                mode='auto')]

generator_mse_model = Generator_MSE_Model(generator_model=generator_model)
generator_mse_model.generator_model.build((1, None, None, 3))
generator_mse_model.compile(generator_optimizer=tf.keras.optimizers.Adam(generator_learning_rate),
                            generator_loss=tf.keras.losses.MeanSquaredError())
generator_mse_model.fit(train_dataset,
                        epochs = srresnet_epochs,
                        callbacks=callbacks)
