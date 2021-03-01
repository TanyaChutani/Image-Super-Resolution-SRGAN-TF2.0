import tensorflow as tf
import argparse
from srgan.models.generator import Generator
from srgan.models.srresnet import Generator_MSE_Model
from srgan.data.data_generator import train_data_generator

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-epoch',
                        '--srresnet_epochs',
                        type=int,
                        metavar='',
                        default = 20000)

    parser.add_argument('-b',
                        '--batch_size',
                        type=int,
                        metavar='',
                        default = 16)

    parser.add_argument('-img_dir',
                        '--train_path',
                        type=str,
                        metavar='',
                        default = '/content/DIV2K_train_HR/*.png')

    parser.add_argument('-w',
                        '--weights_path',
                        type=str,
                        metavar='',
                        default = '/content/srresnet_weights/')

    parser.add_argument('-lr',
                        '--generator_learning_rate',
                        type=int,
                        metavar='',
                        default = 0.0001)

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_dataset = train_data_generator(args.train_path,
                                         args.batch_size)
    generator_mse_model = Generator_MSE_Model(generator_model=Generator())
    generator_mse_model.generator_model.build((1, None, None, 3))
    callbacks = [tf.keras.callbacks.ModelCheckpoint(args.weights_path,
                                                    monitor='mse_loss',
                                                    save_best_only=False,
                                                    save_weights_only=True,
                                                    mode='auto')]

    generator_mse_model = Generator_MSE_Model(generator_model=Generator())
    generator_mse_model.generator_model.build((1, None, None, 3))
    generator_mse_model.compile(generator_optimizer=tf.keras.optimizers.Adam(args.generator_learning_rate),
                                generator_loss=tf.keras.losses.MeanSquaredError())
    generator_mse_model.fit(train_dataset,
                            epochs=args.srresnet_epochs,
                            callbacks=callbacks)



if __name__ == "__main__":
    main()