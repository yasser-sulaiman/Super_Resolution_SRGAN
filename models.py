import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, BatchNormalization, Add, Lambda, LeakyReLU, Flatten, Dense
from tensorflow.python.keras.layers import PReLU

def normalize(x):
    # Normalizes RGB images to [-1, 1].
    return x/127.5 - 1

def normalize_01(x):
    # Normalizes RGB images to [0, 1].
    return x / 255.0

def denormalize(x):
    # denormalizes RGB images from [-1, 1] to [0, 255].
    return (x + 1) * 127.5

def pixel_shuffle(scale):
    return lambda x: tf.nn.depth_to_space(x, scale)

def upsample(x_in, number_Of_filters):
    out = Conv2D(number_Of_filters, kernel_size=3, padding='same')(x_in)
    out = Lambda(pixel_shuffle(scale=2))(out)
    out = PReLU(shared_axes=[1, 2])(out)
    return out

# Descriminator
def discBlock(x_in, number_Of_filters, strides=1, batchnorm=True, momentum=0.8):
    out = Conv2D(number_Of_filters, kernel_size=3, strides=strides, padding='same')(x_in)
    # if Batch Normalization applied add this layer
    if batchnorm:
        out = BatchNormalization(momentum=momentum)(out)
    return LeakyReLU(alpha=0.2)(out)

def discriminatorNet(hr_crop_size):
    x_in = Input(shape=(hr_crop_size, hr_crop_size, 3))
    out = Lambda(normalize)(x_in)

    out = discBlock(out, 64, batchnorm=False)
    out = discBlock(out, 64, strides=2)

    out = discBlock(out, 128)
    out = discBlock(out, 128, strides=2)

    out = discBlock(out, 256)
    out = discBlock(out, 256, strides=2)

    out = discBlock(out, 512)
    out = discBlock(out, 512, strides=2)

    out = Flatten()(out)

    out = Dense(1024)(out)
    out = LeakyReLU(alpha=0.2)(out)
    out = Dense(1, activation='sigmoid')(out)

    return Model(x_in, out)


# Generator
def resBlock(x_in, number_Of_filters, momentum=0.8):
    # Conv2D -> Batch Normalization -> PReLU -> Conv2D -> Batch Normalization -> add skip connection
    out = Conv2D(number_Of_filters, kernel_size=3, padding='same')(x_in)
    out = BatchNormalization(momentum=momentum)(out)
    out = PReLU(shared_axes=[1, 2])(out)
    out = Conv2D(number_Of_filters, kernel_size=3, padding='same')(out)
    out = BatchNormalization(momentum=momentum)(out)
    out = Add()([x_in, out])
    return out

def generatorNet(number_Of_filters=64, number_of_res_blocks=16):

    lowRes = Input(shape=(None, None, 3))
    out = Lambda(normalize_01)(lowRes)

    out = Conv2D(number_Of_filters, kernel_size=9, padding='same')(out)
    out = x_1 = PReLU(shared_axes=[1, 2])(out)

    # add B Residual Blocks
    for _ in range(number_of_res_blocks):
        out = resBlock(out, number_Of_filters)

    out = Conv2D(number_Of_filters, kernel_size=3, padding='same')(out)
    out = BatchNormalization()(out)
    # add the first skip connection
    out = Add()([x_1, out])

    # 4x upsampling
    out = upsample(out, number_Of_filters * 4)
    out = upsample(out, number_Of_filters * 4)

    # tanh because the data are normlized between -1 and 1
    out = Conv2D(3, kernel_size=9, padding='same', activation='tanh')(out)
    sr = Lambda(denormalize)(out)

    return Model(lowRes, sr)