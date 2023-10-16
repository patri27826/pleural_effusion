import os
import cv2
import numpy as np
import pandas as pd
import random
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, DenseNet201
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

label_file_path = 'selected_data.csv' 
df = pd.read_csv(label_file_path)

image_labels = {}  
for index, row in df.iterrows():
    image_labels[row['id']] = row['Fluid Status']

model = DenseNet201(include_top=False, weights='imagenet', input_tensor=None,
               input_shape=(224, 224 ,3))
# 定義輸出層
x = model.output
x = GlobalAveragePooling2D()(x)
predictions = Dense(2, activation='softmax')(x)

model = Model(inputs=model.input, outputs=predictions)
for layer in model.layers[:2]:
    layer.trainable = False
for layer in model.layers[2:]:
    layer.trainable = True
    
# 編譯模型
model.compile(optimizer=Adam(lr=0.00001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 設定模型儲存條件
checkpoint = ModelCheckpoint('Densenet201_checkpoint_v2.h5', verbose=1,
                          monitor='val_loss', save_best_only=True,
                          mode='min')

# 設定lr降低條件(0.001 → 0.0002 → 0.00004 → 0.00001)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                           patience=5, mode='min', verbose=1,
                           min_lr=1e-6)

roi_image_folder = 'roi'  
roi_image_files = os.listdir(roi_image_folder)


random.shuffle(roi_image_files)


total_samples = len(roi_image_files)
train_size = int(0.8 * total_samples)
valid_size = int(0.1 * total_samples)
test_size = total_samples - train_size - valid_size

train_files = ['roi/' + filename for filename in roi_image_files[:train_size]]
valid_files = ['roi/' + filename for filename in roi_image_files[train_size:train_size+valid_size]]
test_files = ['roi/' + filename for filename in roi_image_files[train_size+valid_size:]]

train_labels = [df.loc[df['id'] == int(filename.split('/')[1].split('.')[0]), 'Fluid Status'].values[0] for filename in train_files]
valid_labels = [df.loc[df['id'] == int(filename.split('/')[1].split('.')[0]), 'Fluid Status'].values[0] for filename in valid_files]
test_labels = [df.loc[df['id'] == int(filename.split('/')[1].split('.')[0]), 'Fluid Status'].values[0] for filename in test_files]

train_data = pd.DataFrame({'image_path': train_files, 'label': train_labels})
valid_data = pd.DataFrame({'image_path': valid_files, 'label': valid_labels})
test_data = pd.DataFrame({'image_path': test_files, 'label': test_labels})

batch_size = 2
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

valid_datagen = ImageDataGenerator(rescale=1.0/255.0)
test_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_dataframe(
    train_data,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

for image, label in train_generator:
    print(f"Image shape: {image.shape}, Min value: {np.min(image)}, Max value: {np.max(image)}, Data type: {image.dtype}")
    break

valid_generator = valid_datagen.flow_from_dataframe(
    valid_data,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

test_generator = test_datagen.flow_from_dataframe(
    test_data,
    x_col='image_path',
    y_col='label',
    target_size=(224, 224),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

num_epochs = 400  

history = model.fit_generator(train_generator,
    steps_per_epoch=len(train_generator),
    epochs=num_epochs,
    validation_data=valid_generator,
    callbacks=[checkpoint, reduce_lr])

model.save('./Densenet201_retrained_v2.h5')
print('已儲存Densenet201_retrained_v2.h5')

# 畫出acc學習曲線
acc = history.history['accuracy']
epochs = range(1, len(acc) + 1)
val_acc = history.history['val_accuracy']
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend(loc='lower right')
plt.grid()
# 儲存acc學習曲線
plt.savefig('./acc.png')
plt.show()

# 畫出loss學習曲線
loss = history.history['loss']
val_loss = history.history['val_loss']
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend(loc='upper right')
plt.grid()
# 儲存loss學習曲線
plt.savefig('loss.png')
plt.show()

predictions = model.predict(test_generator)

label_mapping = {'滲出液' : 0, '漏出液' : 1}
test_labels = test_data['label']
test_labels = [label_mapping[label] for label in test_labels]

predicted_labels = np.argmax(predictions, axis=1)
accuracy = accuracy_score(test_labels, predicted_labels)
print(f"Test Accuracy: {accuracy:.2f}")

classification_rep = classification_report(test_labels, predicted_labels)
print("Classification Report:")
print(classification_rep)

confusion_mat = confusion_matrix(test_labels, predicted_labels)
print("Confusion Matrix:")
print(confusion_mat)
