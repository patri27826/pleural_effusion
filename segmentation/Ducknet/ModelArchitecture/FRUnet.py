import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D, BatchNormalization, Activation, Add, Input, Concatenate
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

def residual_block(inputs, filters, strides=1, is_first=False):
    # feature extraction
    if not is_first:
        bn1 = BatchNormalization()(inputs)
        relu1 = Activation("relu")(bn1)
        conv1 = Conv2D(filters, kernel_size=(3, 3), padding="same", strides=strides)(relu1)
    else:
        conv1 = Conv2D(filters, kernel_size=(3, 3), padding="same", strides=strides)(inputs)
    bn2 = BatchNormalization()(conv1)
    relu2 = Activation("relu")(bn2)
    conv2 = Conv2D(filters, kernel_size=(3, 3), padding="same", strides=1)(relu2)

    # shortcut
    shortcut = Conv2D(filters, kernel_size=(1, 1), padding="same", strides=strides)(inputs)
    bn3 = BatchNormalization()(shortcut)

    # addition
    addition = Add()([conv2, bn3])
    return addition

def create_model(img_height, img_width, input_channels, out_classes, starting_filters):
    input_layer = Input((img_height, img_width, input_channels))

    # encode
    residual1 = residual_block(input_layer, starting_filters, 1, True)
    residual2 = residual_block(residual1, starting_filters*2, 2)
    residual3 = residual_block(residual2, starting_filters*4, 2)
    residual4 = residual_block(residual3, starting_filters*8, 2)
    residual5 = residual_block(residual4, starting_filters*16, 2)

    # decode
    up1 = UpSampling2D((2, 2), interpolation=interpolation)(residual5)
    concat1 = Concatenate()([up1, residual4])
    residual6 = residual_block(concat1, starting_filters*8)
    up2 = UpSampling2D((2, 2), interpolation=interpolation)(residual6)
    concat2 = Concatenate()([up2, residual3])
    residual7 = residual_block(concat2, starting_filters*4)
    up3 = UpSampling2D((2, 2), interpolation=interpolation)(residual7)
    concat3 = Concatenate()([up3, residual2])
    residual8 = residual_block(concat3, starting_filters*2)
    up4 = UpSampling2D((2, 2), interpolation=interpolation)(residual8)
    concat4 = Concatenate()([up4, residual1])
    residual9 = residual_block(concat4, starting_filters)

    outputs = Conv2D(out_classes, (1, 1), padding="same", activation="sigmoid")(residual9)
    
    model = Model(inputs=input_layer, outputs=outputs)
    
    return model
