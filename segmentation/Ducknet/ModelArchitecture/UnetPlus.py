import tensorflow as tf
from tensorflow.keras.layers import Conv2D, UpSampling2D, Concatenate
from keras.models import Model

kernel_initializer = 'he_uniform'
interpolation = "nearest"




def standard_unit(inputs, filters):
    conv1 = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(inputs)
    bn1 = tf.keras.layers.BatchNormalization()(conv1)
    conv2 = Conv2D(filters, (3, 3), activation='relu', kernel_initializer='he_normal', padding='same')(bn1)
    bn2 = tf.keras.layers.BatchNormalization()(conv2)
    return bn2

def upsampling_block(inputs, filters):
    upsampling = UpSampling2D((2, 2))(inputs)
    conv = Conv2D(filters, (2, 2), activation='relu', padding='same', kernel_initializer='he_normal')(upsampling)
    bn = tf.keras.layers.BatchNormalization()(conv)
    return bn

def create_model(img_height, img_width, input_channels, out_classes, starting_filters):
    input_layer = tf.keras.layers.Input((img_height, img_width, input_channels))

    # encode
    conv1_1 = standard_unit(input_layer, starting_filters)
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1_1)
    conv2_1 = standard_unit(pool1, starting_filters * 2)
    pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2_1)
    conv3_1 = standard_unit(pool2, starting_filters * 4)
    pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3_1)
    conv4_1 = standard_unit(pool3, starting_filters * 8)
    pool4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4_1)
    conv5_1 = standard_unit(pool4, starting_filters * 16)

    # skip connection
    up1_2 = upsampling_block(conv2_1, starting_filters * 2)
    concat1 = Concatenate()([up1_2, conv1_1])
    conv1_2 = standard_unit(concat1, starting_filters)
    up2_2 = upsampling_block(conv3_1, starting_filters * 4)
    concat2 = Concatenate()([up2_2, conv2_1])
    conv2_2 = standard_unit(concat2, starting_filters * 2)
    up1_3 = upsampling_block(conv2_2, starting_filters * 2)
    concat3 = Concatenate()([up1_3, conv1_1, conv1_2])
    conv1_3 = standard_unit(concat3, starting_filters)
    up3_2 = upsampling_block(conv4_1, starting_filters * 8)
    concat4 = Concatenate()([up3_2, conv3_1])
    conv3_2 = standard_unit(concat4, starting_filters * 4)
    up2_3 = upsampling_block(conv3_2, starting_filters * 4)
    concat5 = Concatenate()([up2_3, conv2_1, conv2_2])
    conv2_3 = standard_unit(concat5, starting_filters * 2)
    up1_4 = upsampling_block(conv2_3, starting_filters * 2)
    concat6 = Concatenate()([up1_4, conv1_1, conv1_2, conv1_3])
    conv1_4 = standard_unit(concat6, starting_filters)

    # decode
    up4_2 = upsampling_block(conv5_1, starting_filters * 8)
    concat7 = Concatenate()([up4_2, conv4_1])
    conv4_2 = standard_unit(concat7, starting_filters * 8)
    up3_3 = upsampling_block(conv4_2, starting_filters * 4)
    concat8 = Concatenate()([up3_3, conv3_1, conv3_2])
    conv3_3 = standard_unit(concat8, starting_filters * 4)
    up2_4 = upsampling_block(conv3_3, starting_filters * 2)
    concat9 = Concatenate()([up2_4, conv2_1, conv2_2, conv2_3])
    conv2_4 = standard_unit(concat9, starting_filters * 2)
    up1_5 = upsampling_block(conv2_4, starting_filters)
    concat10 = Concatenate()([up1_5, conv1_1, conv1_2, conv1_3, conv1_4])
    conv1_5 = standard_unit(concat10, starting_filters)

    output = Conv2D(out_classes, (1, 1), activation='sigmoid')(conv1_5)

    model = Model(inputs=input_layer, outputs=output)

    return model
