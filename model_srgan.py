import tensorflow as tf
import argparse
from srgan.models.generator import Generator
from srgan.models.discriminator import Discriminator
from srgan.models.srresnet import Generator_MSE_Model
from srgan.data.data_generator import train_data_generator
from srgan.loss.psnr_loss import psnr_loss
from srgan.models.srgan import SRGAN

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gan_epoch',
                        '--srgan_epochs',
                        type=int,
                        metavar='',
                        default = 4000)

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
                        '--gan_weights_path',
                        type=str,
                        metavar='',
                        default = '/content/srgan_weights/')



    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    train_dataset = train_data_generator(args.train_path,
                                         args.batch_size)


    learning_rate = tf.optimizers.schedules.PiecewiseConstantDecay(boundaries=[args.srgan_epochs / 2], values=[0.0001, 0.00001])

    srgan = SRGAN(generator_model=Generator(),
                  discriminator_model=Discriminator())
    srgan.generator_model.build((1,None,None,3))
    test_image_height = 1300
    test_image_width = 2000
    channels = 3
    srgan.discriminator_model.build((1,test_image_height, test_image_width, channels))
    step_per_epoch = len(train_dataset)
    callbacks = [tf.keras.callbacks.ModelCheckpoint(args.gan_weights_path,monitor='loss',save_best_only=False,save_weights_only=True, mode='auto')]
    srgan.compile(generator_optimizer = tf.keras.optimizers.Adam(learning_rate),
                  discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate),
                  loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True,reduction='none'),
                  srgan_metrics=psnr_loss)
    srgan.fit(train_dataset,epochs=(args.srgan_epochs),
              callbacks = callbacks)

if __name__ == "__main__":
    main()
