import numpy as np 
import os
import skimage.io as io
import skimage.transform as trans
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
#from tensorflow.contrib.opt import AdamWOptimizer
#from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras import backend as K
import sklearn.metrics as sm
from tensorflow.keras.layers import BatchNormalization as BN
from tensorflow.python.keras.utils.data_utils import get_file
import tensorflow as tf


learning_rate = 1e-4
learning_decay_rate = 0.00001
img_size = (256,256,1) # 256 * 256 grayscale img with 1 channel
dr_rate = 0.6 # never mind
leakyrelu_alpha = 0.3

def balanced_cross_entropy(y_true, y_pred):
    beta = tf.reduce_mean(1 - y_true)
    weight_a = beta * tf.cast(y_true, tf.float32)
    weight_b = (1 - beta) * tf.cast(1 - y_true, tf.float32)
    o = (tf.math.log1p(tf.exp(-tf.abs(y_pred))) + tf.nn.relu(-y_pred)) * (weight_a + weight_b) + y_pred * weight_b
    return tf.reduce_mean(o)


def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


def BCDU_net_D3(input_size = (256,256,1)):
    N = input_size[0]
    inputs = Input(input_size) 
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
  
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    drop3 = Dropout(0.5)(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    # D1
    conv4 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)     
    conv4_1 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4_1 = Dropout(0.5)(conv4_1)
    # D2
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(drop4_1)     
    conv4_2 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_2)
    conv4_2 = Dropout(0.5)(conv4_2)
    # D3
    merge_dense = concatenate([conv4_2,drop4_1], axis = 3)
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge_dense)     
    conv4_3 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4_3)
    drop4_3 = Dropout(0.5)(conv4_3)
    
    up6 = Conv2DTranspose(256, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(drop4_3)
    up6 = BatchNormalization(axis=3)(up6)
    up6 = Activation('relu')(up6)

    x1 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(drop3)
    x2 = Reshape(target_shape=(1, np.int32(N/4), np.int32(N/4), 256))(up6)
    merge6  = concatenate([x1,x2], axis = 1) 
    merge6 = ConvLSTM2D(filters = 128, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge6)
            
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)

    up7 = Conv2DTranspose(128, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv6)
    up7 = BatchNormalization(axis=3)(up7)
    up7 = Activation('relu')(up7)

    x1 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(conv2)
    x2 = Reshape(target_shape=(1, np.int32(N/2), np.int32(N/2), 128))(up7)
    merge7  = concatenate([x1,x2], axis = 1) 
    merge7 = ConvLSTM2D(filters = 64, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge7)
        
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)

    up8 = Conv2DTranspose(64, kernel_size=2, strides=2, padding='same',kernel_initializer = 'he_normal')(conv7)
    up8 = BatchNormalization(axis=3)(up8)
    up8 = Activation('relu')(up8)    

    x1 = Reshape(target_shape=(1, N, N, 64))(conv1)
    x2 = Reshape(target_shape=(1, N, N, 64))(up8)
    merge8  = concatenate([x1,x2], axis = 1) 
    merge8 = ConvLSTM2D(filters = 32, kernel_size=(3, 3), padding='same', return_sequences = False, go_backwards = True,kernel_initializer = 'he_normal' )(merge8)    
    
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv8 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    conv9 = Conv2D(1, 1, activation = 'sigmoid')(conv8)

    model = Model(inputs, conv9)
    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])    
    return model


def mlti_res_block(inputs, filter_size1, filter_size2, filter_size3, filter_size4):
    cnn1 = Conv2D(filter_size1, (3, 3), padding='same', activation="relu")(inputs)
    cnn2 = Conv2D(filter_size2, (3, 3), padding='same', activation="relu")(cnn1)
    cnn3 = Conv2D(filter_size3, (3, 3), padding='same', activation="relu")(cnn2)

    cnn = Conv2D(filter_size4, (1, 1), padding='same', activation="relu")(inputs)

    concat = Concatenate()([cnn1, cnn2, cnn3])
    add = Add()([concat, cnn])

    return add

def res_path(inputs, filter_size, path_number):
    def block(x, fl):
        cnn1 = Conv2D(filter_size, (3, 3), padding='same', activation="relu")(inputs)
        cnn2 = Conv2D(filter_size, (1, 1), padding='same', activation="relu")(inputs)

        add = Add()([cnn1, cnn2])

        return add

    cnn = block(inputs, filter_size)
    if path_number <= 3:
        cnn = block(cnn, filter_size)
        if path_number <= 2:
            cnn = block(cnn, filter_size)
            if path_number <= 1:
                cnn = block(cnn, filter_size)

    return cnn


def multi_res_u_net(pretrained_weights=None, input_size=(256, 256, 1)):
    inputs = Input(input_size)

    res_block1 = mlti_res_block(inputs, 8, 17, 26, 51)
    pool1 = MaxPool2D()(res_block1)

    res_block2 = mlti_res_block(pool1, 17, 35, 53, 105)
    pool2 = MaxPool2D()(res_block2)

    res_block3 = mlti_res_block(pool2, 31, 72, 106, 209)
    pool3 = MaxPool2D()(res_block3)

    res_block4 = mlti_res_block(pool3, 71, 142, 213, 426)
    pool4 = MaxPool2D()(res_block4)

    res_block5 = mlti_res_block(pool4, 142, 284, 427, 853)
    upsample = UpSampling2D()(res_block5)

    res_path4 = res_path(res_block4, 256, 4)
    concat = Concatenate()([upsample, res_path4])

    res_block6 = mlti_res_block(concat, 71, 142, 213, 426)
    upsample = UpSampling2D()(res_block6)

    res_path3 = res_path(res_block3, 128, 3)
    concat = Concatenate()([upsample, res_path3])

    res_block7 = mlti_res_block(concat, 31, 72, 106, 209)
    upsample = UpSampling2D()(res_block7)

    res_path2 = res_path(res_block2, 64, 2)
    concat = Concatenate()([upsample, res_path2])

    res_block8 = mlti_res_block(concat, 17, 35, 53, 105)
    upsample = UpSampling2D()(res_block8)

    res_path1 = res_path(res_block1, 32, 1)
    concat = Concatenate()([upsample, res_path1])

    res_block9 = mlti_res_block(concat, 8, 17, 26, 51)
    sigmoid = Conv2D(1, (1, 1), padding='same', activation="sigmoid")(res_block9)

    model = Model(inputs, sigmoid)
    model.compile(optimizer = Adam(learning_rate = 1e-4,decay = learning_decay_rate), loss = 'binary_crossentropy', metrics = ['accuracy'])

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model