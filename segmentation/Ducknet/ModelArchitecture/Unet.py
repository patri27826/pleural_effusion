import tensorflow as tf
import tensorflow.keras.layers as layers
import numpy as np

kernel_initializer = 'he_uniform'
interpolation = "nearest"


def create_model(img_height, img_width, input_channels, out_classes, starting_filters):
    input_layer = tf.keras.layers.Input((img_height, img_width, input_channels))

    print('Starting UNet')

    # Encode
    conv1 = standard_unit(input_layer, starting_filters)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)
    conv2 = standard_unit(pool1, starting_filters * 2)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)
    conv3 = standard_unit(pool2, starting_filters * 4)
    pool3 = layers.MaxPooling2D((2, 2))(conv3)
    conv4 = standard_unit(pool3, starting_filters * 8)
    pool4 = layers.MaxPooling2D((2, 2))(conv4)
    conv5 = standard_unit(pool4, starting_filters * 16)

    # Decode
    up1 = upsampling_block(conv5, starting_filters * 8)
    concat1 = layers.Concatenate()([conv4, up1])
    conv6 = standard_unit(concat1, starting_filters * 8)
    up2 = upsampling_block(conv6, starting_filters * 4)
    concat2 = layers.Concatenate()([conv3, up2])
    conv7 = standard_unit(concat2, starting_filters * 4)
    up3 = upsampling_block(conv7, starting_filters * 2)
    concat3 = layers.Concatenate()([conv2, up3])
    conv8 = standard_unit(concat3, starting_filters * 2)
    up4 = upsampling_block(conv8, starting_filters)
    concat4 = layers.Concatenate()([conv1, up4])
    conv9 = standard_unit(concat4, starting_filters)

    outputs = layers.Conv2D(out_classes, (1, 1), activation='sigmoid')(conv9)

    model = tf.keras.models.Model(inputs=input_layer, outputs=outputs)

    return model

def standard_unit(inputs, filters):
    conv1 = layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    bn1 = layers.BatchNormalization()(conv1)
    conv2 = layers.Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn1)
    bn2 = layers.BatchNormalization()(conv2)
    return bn2

def upsampling_block(inputs, filters):
    upsampling = layers.UpSampling2D((2, 2))(inputs)
    conv = layers.Conv2D(filters, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(upsampling)
    bn = layers.BatchNormalization()(conv)
    return bn
