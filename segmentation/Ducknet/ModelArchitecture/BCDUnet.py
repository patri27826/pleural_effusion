import tensorflow as tf
from keras.layers import Conv2D, UpSampling2D
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

def BConvLSTM(in1, in2, d, fi, fo):
    input_size = (352, 352, 1)
    x1 = tf.keras.layers.Reshape(target_shape=(1, int(input_size[0]/d), int(input_size[1]/d), fi))(in1)
    x2 = tf.keras.layers.Reshape(target_shape=(1, int(input_size[0]/d), int(input_size[1]/d), fi))(in2)
    merge = tf.keras.layers.concatenate([x1, x2], axis=1) 
    merge = tf.keras.layers.ConvLSTM2D(fo, (3, 3), padding='same', return_sequences=False, go_backwards=True, kernel_initializer='he_normal')(merge)
    return merge

def create_model(img_height, img_width, input_channels, out_classes, starting_filters):
    filters = [16, 32, 64, 128, 256]
    input_layer = tf.keras.layers.Input((img_height, img_width, input_channels))

    # encode
    conv1 = standard_unit(input_layer, filters[0])
    pool1 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv1)
    conv2 = standard_unit(pool1, filters[1])
    pool2 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv2)
    conv3 = standard_unit(pool2, filters[2])
    pool3 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv3)
    conv4 = standard_unit(pool3, filters[3])
    pool4 = tf.keras.layers.MaxPooling2D((2, 2), strides=(2, 2))(conv4)

    # D1
    conv5_1 = standard_unit(pool4, filters[4])
    # D2
    conv5_2 = standard_unit(conv5_1, filters[4])
    # D3
    merge_dense = tf.keras.layers.concatenate([conv5_2, conv5_1], axis=3)
    conv5_3 = standard_unit(merge_dense, filters[4])

    # decode
    up1 = upsampling_block(conv5_3, filters[3])
    LSTM1 = BConvLSTM(conv4, up1, 8, filters[3], filters[2])
    conv6 = standard_unit(LSTM1, filters[3])
    up2 = upsampling_block(conv6, filters[2])
    LSTM2 = BConvLSTM(conv3, up2, 4, filters[2], filters[1])
    conv7 = standard_unit(LSTM2, filters[2])
    up3 = upsampling_block(conv7, filters[1])
    LSTM3 = BConvLSTM(conv2, up3, 2, filters[1], filters[0])
    conv8 = standard_unit(LSTM3, filters[1])
    up4 = upsampling_block(conv8, filters[0])
    LSTM4 = BConvLSTM(conv1, up4, 1, filters[0], int(filters[0]/2))
    conv9 = standard_unit(LSTM4, filters[0])

    outputs = Conv2D(out_classes, (1, 1), activation='sigmoid')(conv9)

    model = Model(inputs=input_layer, outputs=outputs)

    return model
