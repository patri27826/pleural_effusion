import os
import random
import timeit
import cv2
import datetime
import matplotlib.pyplot as plt
import tensorflow.keras
import numpy as np
from sklearn.metrics import jaccard_score
from keras.preprocessing.image import ImageDataGenerator
from UNet import *
from model import *

class DataGen(tensorflow.keras.utils.Sequence):
    def __init__(self, ids, path, batch_size=8, image_size=(256, 256)):
        self.ids = ids
        self.path = path
        self.batch_size = batch_size
        self.image_size = image_size
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.ids) / float(self.batch_size)))

    def load(self, name):
        image_path = os.path.join(self.path, "image_resize/", name)
        mask_path = os.path.join(self.path, "label_resize/", name)
        #print(name, image_path)
        try:
            image = cv2.imread(image_path)
            image = cv2.resize(image, (256,256), interpolation=cv2.INTER_NEAREST)
        except:
            print('wrong',image_path)
        try:
            mask = cv2.imread(mask_path, 0)
            mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)
            mask = np.expand_dims(mask, axis=-1)
        except:
            print('wrong',mask_path)
        
        # mask = cv2.imread(mask_path, 0)
        # mask = cv2.resize(mask, (256,256), interpolation=cv2.INTER_NEAREST)
        # mask = np.expand_dims(mask, axis=-1)
        

        image = image / 255.0
        mask = mask / 255.0
        return image, mask
    
    def __getitem__(self, index):
        if (index + 1) * self.batch_size > len(self.ids):
            files = self.ids[index * self.batch_size:]
        else:
            files = self.ids[index * self.batch_size:(index + 1) * self.batch_size]

        print('start', files)
        # print(len(files))
        images = []
        masks  = []
        for name in files:
            image, mask = self.load(name)
            images.append(image)
            masks.append(mask)
            
        images = np.array(images)
        masks  = np.array(masks)
        return images, masks
    
    def on_epoch_end(self):
        pass

def check_image(train_ids, train_path, batch_size, image_size):
    gen = DataGen(train_ids, train_path, batch_size=batch_size, image_size=image_size)
    x, y = gen.__getitem__(0)
    print(x.shape, y.shape)
    r = random.randint(0, len(x) - 1)
    fig = plt.figure()
    fig.subplots_adjust(hspace=0.4, wspace=0.4)
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(x[r])
    ax = fig.add_subplot(1, 2, 2)
    ax.imshow(np.reshape(y[r] * 255, image_size), cmap='gray')
    plt.show()

def postproccessing(image):
    kernels = [5, 10, 15]
    for k in kernels: 
        kernel = np.ones((k,k), np.uint8)
        # opening
        erosion1 = cv2.erode(image, kernel, iterations = 1)
        dilation1 = cv2.dilate(erosion1, kernel, iterations = 1)

        # closing
        dilation2 = cv2.dilate(dilation1, kernel, iterations = 1)
        erosion2 = cv2.erode(dilation2, kernel, iterations = 1)
        
        image = erosion2
    

    return image

def calculate_dice_coefficient(folder_name):
    index = os.listdir(os.path.join(os.getcwd(), 'results', folder_name, 'image'))
    mask_index = os.listdir(os.path.join(os.getcwd(), 'results', folder_name, 'label'))
    dice_coefficients = []
    
    for i in range(len(index)):
        img_true = cv2.imread(os.path.join(os.getcwd(), 'results', folder_name, 'label', mask_index[i]))
        img_true = cv2.cvtColor(img_true, cv2.COLOR_BGR2GRAY)
        img_true[img_true < 128] = 0
        img_true[img_true >= 128] = 1
        
        img_pred = cv2.imread(os.path.join(os.getcwd(), 'results', folder_name, 'results', index[i]))
        img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2GRAY)
        img_pred[img_pred < 128] = 0
        img_pred[img_pred >= 128] = 1
        
        dice = dice_coefficient(img_true, img_pred)
        dice_coefficients.append(dice)
    
    return dice_coefficients

def dice_coefficient(y_true, y_pred):
    intersection = np.sum(y_true * y_pred)
    union = np.sum(y_true) + np.sum(y_pred)
    dice = (2.0 * intersection) / (union + 1e-7)  # 添加一個小數以避免除以零的錯誤
    return dice

def calculate_jaccard_index(folder_name):
    index = os.listdir(f'{os.getcwd()}/results/' + folder_name + '/image/')
    mask_index = os.listdir(f'{os.getcwd()}/results/' + folder_name + '/label/')
    iou = []
    for i in range(len(index)):
        img_true = cv2.imread(f'{os.getcwd()}/results/' + folder_name + '/label/' + mask_index[i], 0)
        img_true[img_true < 128] = 0
        img_true[img_true >= 128] = 1
        img_pred = cv2.imread(f'{os.getcwd()}/results/' + folder_name + '/results/' + index[i], 0)
        img_pred[img_pred < 128] = 0
        img_pred[img_pred >= 128] = 1
        img_true = np.array(img_true).ravel()
        img_pred = np.array(img_pred).ravel()
        ji = jaccard_score(img_true, img_pred)
        iou.append(ji)

        # if i == '425.png':
        #     print(folder_name + ': ' + str(ji))

    return iou

def save_figure(history, feature):
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model dice coefficient')
    plt.ylabel('Dice')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f'{os.getcwd()}/figures/' + feature + '_dice.png')
    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.savefig(f'{os.getcwd()}/figures/' + feature + '_loss.png')
    plt.clf()

def save_results(model, image_size, folder_name):
    if not os.path.exists(f'{os.getcwd()}/results/' + folder_name):
        os.mkdir(f'{os.getcwd()}/results/' + folder_name)
        os.mkdir(f'{os.getcwd()}/results/' + folder_name + '/image')
        os.mkdir(f'{os.getcwd()}/results/' + folder_name + '/label')
        os.mkdir(f'{os.getcwd()}/results/' + folder_name + '/results')
        os.mkdir(f'{os.getcwd()}/results/' + folder_name + '/o_results')

    inference_time = []
    test_ids = os.listdir(f'{os.getcwd()}/data/test/image_resize/')
    test_mask_ids = os.listdir(f'{os.getcwd()}/data/test/label_resize/')
    for i in range(len(test_ids)):
        
        x = cv2.imread(f'{os.getcwd()}/data/test/image_resize/' + test_ids[i])
        cv2.imwrite(f'{os.getcwd()}/results/' + folder_name + '/image/' + test_ids[i], x)
        
        y = cv2.imread(f'{os.getcwd()}/data/test/label_resize/' + test_mask_ids[i])
        cv2.imwrite(f'{os.getcwd()}/results/' + folder_name + '/label/' + test_mask_ids[i], y)
        size = np.shape(x)

        x = cv2.resize(x, (608, 608))
        x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
        x = x / 255.0
        x = np.expand_dims(x, axis=0)

        start = timeit.default_timer()
        results = model.predict(x)
        results = results >= 0.5
        stop = timeit.default_timer()
        inference_time.append(round(stop - start, 2))
        results = np.reshape(results * 255, image_size)
        results = np.stack((results,) * 3, -1)
        results = results.astype(np.uint8)
        results = cv2.cvtColor(results, cv2.COLOR_BGR2GRAY)
        o_results = cv2.resize(results, (size[1], size[0]))
        cv2.imwrite(f'{os.getcwd()}/results/' + folder_name + '/o_results/' + test_ids[i], o_results)
        results = postproccessing(results)
        results = cv2.resize(results, (size[1], size[0]))
        cv2.imwrite(f'{os.getcwd()}/results/' + folder_name + '/results/' + test_ids[i], results)

    return round(sum(inference_time) / len(inference_time), 2)

def train(model, feature, image_size, batch_size, epochs, train_gen, train_steps, valid_gen, valid_steps):
    reduce_lr_loss = tensorflow.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', mode='min', factor=0.1, patience=8, verbose=1)
    checkpoint = tensorflow.keras.callbacks.ModelCheckpoint(filepath=f'{os.getcwd()}/models/' + feature + '.h5', monitor='val_dice_coef', mode='max', save_best_only=True, save_weights_only=True)

    start = timeit.default_timer()
    history = model.fit(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=epochs, verbose=1, callbacks=[checkpoint, reduce_lr_loss])
    stop = timeit.default_timer()
    train_time = round(stop - start, 2)
    save_figure(history, feature)

    model.load_weights(f'{os.getcwd()}/models/' + feature + '.h5')
    inference_time = save_results(model, image_size, feature)
    iou = calculate_jaccard_index(feature)
    dice = calculate_dice_coefficient(feature)
    return train_time, inference_time, iou, dice

def run(image_size, batch_size, epochs, filters, learning_rate, train_gen, train_steps, valid_gen, valid_steps):
    feature = '_' + 'fullimage' +'_' + 'aug_v1' + '_' + str(datetime.date.today()) + '_' + str(image_size) + '_' + str(batch_size) + '_' + str(epochs) + '_' + str(learning_rate) + '_' + str(filters[0])

    myBCDUNet = BCDUNet(image_size, learning_rate)
    model1 = myBCDUNet.build_model(filters)
    #model1.summary()
    training_time1, inference_time1, iou1, dice1 = train(model1, 'BCDUNet' + feature, image_size, batch_size, epochs, train_gen, train_steps, valid_gen, valid_steps)
    
    # myFRUNet = FRUNet(image_size, learning_rate)
    # model2 = myFRUNet.build_model(filters)
    # #model2.summary()
    # training_time2, inference_time2, iou2, dice2 = train(model2, 'FRUNet' + feature, image_size, batch_size, epochs, train_gen, train_steps, valid_gen, valid_steps)

    # myUNetPlusPlus = UNetPlusPlus(image_size, learning_rate)
    # model3 = myUNetPlusPlus.build_model(filters)
    # #model3.summary()
    # training_time3, inference_time3, iou3, dice3 = train(model3, 'UNet++' + feature, image_size, batch_size, epochs, train_gen, train_steps, valid_gen, valid_steps)

    # myUNet = UNet(image_size, learning_rate)
    # model4 = myUNet.build_model(filters)
    # #model4.summary()
    # training_time4, inference_time4, iou4, dice4 = train(model4, 'UNet' + feature, image_size, batch_size, epochs, train_gen, train_steps, valid_gen, valid_steps)

    
    f = open(f'{os.getcwd()}/fullimg_summary_{str(datetime.date.today())}.txt', 'a')
    lines = ['----------Summary' + feature + '----------\n',
             'Accuracy(Jaccard index)\n',
             'BCDUNet: ' + str(round(sum(iou1) / len(iou1), 4)) + ' +- ' + str(round(np.var(iou1), 4)) + '\n',
            #  'FRUNet: '  + str(round(sum(iou2) / len(iou2), 4)) + ' +- ' + str(round(np.var(iou2), 4)) + '\n',
            #  'UNet++: '  + str(round(sum(iou3) / len(iou3), 4)) + ' +- ' + str(round(np.var(iou3), 4)) + '\n',
            #  'UNet: '    + str(round(sum(iou4) / len(iou4), 4)) + ' +- ' + str(round(np.var(iou4), 4)) + '\n\n',
             'Dice Coefficient\n',
             'BCDUNet: ' + str(np.mean(dice1)) + '\n',
            #  'FRUNet: '  + str(np.mean(dice2)) + '\n',
            #  'UNet++: '  + str(np.mean(dice3)) + '\n',
            #  'UNet: '    + str(np.mean(dice4)) + '\n\n',
             'Training Time\n',
             'BCDUNet: ' + str(training_time1) + '\n',
            #  'FRUNet: '  + str(training_time2) + '\n',
            #  'UNet++: '  + str(training_time3) + '\n',
            #  'UNet: '    + str(training_time4) + '\n\n',
             'Inference Time\n',
             'BCDUNet: ' + str(inference_time1) + '\n',
            #  'FRUNet: '  + str(inference_time2) + '\n',
            #  'UNet++: '  + str(inference_time3) + '\n',
            #  'UNet: '    + str(inference_time4) + '\n',
             '------------------------------------------\n\n']
    f.writelines(lines)
    f.close()
    
def adjust(zip_generator):
    for (img, mask) in zip_generator:
        mask = normalize_mask(mask)#二值化
        yield (img, mask)

def normalize_mask(mask):
    """ Mask Normalization
    Function that returns normalized mask
    Each pixel is either 0 or 1
    """
    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0
    return mask

def Generator(batch, path, img_size):
    
    SEED = 100
    """training Imagedatagenerator"""
    train_image_datagen = ImageDataGenerator( #**data_gen_args)
                    rescale=1. / 255,
                    width_shift_range = 0.1,
                    height_shift_range = 0.1,
                    # brightness_range = [0.8, 1.2],
                    shear_range = 0.1,
                    rotation_range = 20,
                    fill_mode='nearest',
                    horizontal_flip=True,
                    # vertical_flip=True,
                    zoom_range = 0.1
    )
    train_mask_datagen = ImageDataGenerator( #**data_gen_args)
                    rescale=1. / 255,
                    width_shift_range = 0.1,
                    height_shift_range = 0.1,
                    # brightness_range = [0.8, 1.2],
                    shear_range = 0.1,
                    rotation_range = 20,
                    fill_mode='nearest',
                    horizontal_flip=True,
                    # vertical_flip=True,
                    zoom_range = 0.1
    )
    train_image_generator = train_image_datagen.flow_from_directory(
                    os.path.join(path, 'train'),
                    classes = ['image_resize'],
                    color_mode = 'grayscale',
                    target_size = img_size,
                    # interpolation='nearest',
                    class_mode=None,
                    batch_size = batch,
                    seed = SEED,
                    shuffle = True
    )
    train_mask_generator = train_mask_datagen.flow_from_directory(
                    os.path.join(path, 'train'),
                    classes = ['label_resize'],
                    color_mode = 'grayscale',
                    target_size = img_size,
                    # interpolation='nearest',
                    class_mode=None,
                    batch_size = batch,
                    seed = SEED,
                    shuffle = True
    )
    
    """validation Imagedatagenerator"""
    valid_image_datagen = ImageDataGenerator( rescale=1. / 255)
    valid_mask_datagen = ImageDataGenerator( rescale=1. / 255)
    valid_image_generator = valid_image_datagen.flow_from_directory(
                    os.path.join(path, 'valid'),
                    classes = ['image_resize'],
                    color_mode = 'grayscale',
                    target_size = img_size,
                    # interpolation='nearest',
                    class_mode=None,
                    batch_size = batch,
                    seed = SEED
    )
    valid_mask_generator = valid_mask_datagen.flow_from_directory(
                    os.path.join(path, 'valid'),
                    classes = ['label_resize'],
                    color_mode = 'grayscale',
                    target_size = img_size,
                    # interpolation='nearest',
                    class_mode=None,
                    batch_size = batch,
                    seed = SEED
    )
    # print('here', image_generator[0][0].shape)
    # for each in mask_generator:
    #     print(each[0].shape)
    #     print(each[1].shape)
    #     break
    train_zip_generator = zip(train_image_generator, train_mask_generator)
    valid_zip_generator = zip(valid_image_generator, valid_mask_generator)
    # for (img, mask) in train_zip_generator:
    #     mask = normalize_mask(mask)#二值化
    #     yield (img, mask)
    return train_zip_generator, valid_zip_generator


if __name__ == '__main__':
    seed = 55688
    # random.seed = seed
    np.random.seed(seed)
    tf.seed = seed

    os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

    if not os.path.exists('models'):
        os.mkdir('models')

    if not os.path.exists('results'):
        os.mkdir('results')

    if not os.path.exists('figures'):
        os.mkdir('figures')

    # dataset_path = 'C:/Users/Jeff/Desktop/jiajun_project/model/data/'
    # train_path = 'C:/Users/Jeff/Desktop/jiajun_project/model/data/train/'
    # valid_path = 'C:/Users/Jeff/Desktop/jiajun_project/model/data/valid/'
    # train_ids = os.listdir(train_path + 'image/')
    # valid_ids = os.listdir(valid_path + 'image/')

    # #image_size = (256, 512)
    # image_size = (256, 256)
    # epoch = 50
    # filters = [16, 32, 64, 128, 256]
    # learning_rate = 0.0001


    # dataset_path = 'C:/Users/user/Desktop/jiajun_project/code/Fout_net/data/'
    
    # train_path = dataset_path + 'train/'
    # train_ids = os.listdir(train_path + 'image_resize/')
    
    # valid_path = dataset_path + 'valid/'
    # valid_ids = os.listdir(valid_path + 'image_resize/')
    
    # image_size = (256, 256)
    # epoch = 100
    # filters = [16, 32, 64, 128, 256]
    # learning_rate = 1e-4
    # val_data_size = 67
    
    # valid_ids = train_ids[:val_data_size]
    # train_ids = train_ids[val_data_size:]
    #check_image(train_ids, train_path, batch_size, image_size)

    # batch_size = 8
    # check_image(train_ids, train_path, batch_size, image_size)00
    # time = 5
    # count = 4
    
    # while count < time:
    #     # batch_size = 2
    #     batch_size = 2
    #     train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
    #     print(train_gen[0][0].shape)
    #     train_steps = len(train_ids)//batch_size
    #     valid_gen = DataGen(valid_ids, valid_path, image_size=image_size, batch_size=batch_size)
    #     valid_steps = len(valid_ids)//batch_size
    #     run(image_size, batch_size, epoch, filters, learning_rate, train_gen, train_steps, valid_gen, valid_steps, count)

    #     # batch_size = 4
    #     # batch_size = 4
    #     # train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
    #     # train_steps = len(train_ids)//batch_size
    #     # valid_gen = DataGen(valid_ids, valid_path, image_size=image_size, batch_size=batch_size)
    #     # valid_steps = len(valid_ids)//batch_size
    #     # run(image_size, batch_size, epoch, filters, learning_rate, train_gen, train_steps, valid_gen, valid_steps, count)

    #     # # batch size = 8
    #     # batch_size = 8
    #     # train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
    #     # train_steps = len(train_ids)//batch_size
    #     # valid_gen = DataGen(valid_ids, valid_path, image_size=image_size, batch_size=batch_size)
    #     # valid_steps = len(valid_ids)//batch_size
    #     # run(image_size, batch_size, epoch, filters, learning_rate, train_gen, train_steps, valid_gen, valid_steps, count)
        
    #     # batch_size = 16
    #     # train_gen = DataGen(train_ids, train_path, image_size=image_size, batch_size=batch_size)
    #     # train_steps = len(train_ids)//batch_size
    #     # valid_gen = DataGen(valid_ids, valid_path, image_size=image_size, batch_size=batch_size)
    #     # valid_steps = len(valid_ids)//batch_size
    #     # run(image_size, batch_size, epoch, filters, learning_rate, train_gen, train_steps, valid_gen, valid_steps, count)

    #     count += 1
    
    """Augmentation"""
    dataset_path = f'{os.getcwd()}/data/'
    
    train_path = os.path.join(dataset_path,'train')
    valid_path = os.path.join(dataset_path,'valid')
    
    epoch = 50
    filters = [16, 32, 64, 128, 256]
    # learning_rate = [1e-3, 1e-4]
    learning_rate = [1e-4]
    # Batch = [2, 4]
    Batch = [4]
    input_size = (608, 608)
    batch_size = 8

    for B in Batch:
        for l in learning_rate:
            # 產生training validation data
            train_gen, valid_gen = Generator(
                batch = B,
                path = dataset_path,
                img_size = input_size,
            )
            train_gen, valid_gen = adjust(train_gen), adjust(valid_gen)

            train_steps = 276 // B
            valid_steps = 35 // B
            print(train_steps, valid_steps)
            run(input_size, B, epoch, filters, l, train_gen, train_steps, valid_gen, valid_steps)
                # myUNet = UNet(input_size, learning_rate)
                # model4 = myUNet.build_model(filters)
                # # model = BCDU_net_D3()
                # print(model4.summary)
                # history = model4.fit_generator(train_gen, validation_data=valid_gen, steps_per_epoch=train_steps, validation_steps=valid_steps, epochs=5, verbose=1)

