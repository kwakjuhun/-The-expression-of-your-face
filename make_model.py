"""
in colab

당신의 웃음은?
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import numpy as np
import matplotlib.pyplot as plt

from google.colab import drive
drive.mount('/content/gdrive/')


!mkdir -p ./data
!unzip -uq gdrive/'My Drive'/data.zip -d ./data
!ls data/data/val_dir/


train_dir = os.path.join('./data/train_dir')
x_train_dir = os.path.join(train_dir,'yes')
y_train_dir = os.path.join(train_dir,'no')
val_dir = os.path.join('./data/val_dir')
x_val_dir = os.path.join(val_dir,'yes')
y_val_dir = os.path.join(val_dir,'no')

x_train_size = len(os.listdir(x_train_dir))
y_train_size = len(os.listdir(y_train_dir))
train_size = x_train_size + y_train_size
x_val_size = len(os.listdir(x_val_dir))
y_val_size = len(os.listdir(y_val_dir))
val_size = x_val_size + y_val_size

BATCH_SIZE = 30
IMG_SHAPE = 150

print(train_size, val_size)


# 이미지 데이터 생성기
image_gen_train = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

# 생성기로 이미지 투투두두둑
train_data_gen = image_gen_train.flow_from_directory(batch_size=BATCH_SIZE,
                                                     directory=train_dir,
                                                     color_mode='grayscale',
                                                     shuffle=True,
                                                     target_size=(IMG_SHAPE,IMG_SHAPE),
                                                     class_mode='binary')


# 원본 데이터 유지
# 모델이 훈련 된 후 표시되는 이미지를 반영시키기 위해
image_gen_val = ImageDataGenerator(rescale=1./255)

val_data_gen = image_gen_val.flow_from_directory(batch_size=BATCH_SIZE,
                                                 directory=val_dir,
                                                 color_mode='grayscale',
                                                 target_size=(IMG_SHAPE, IMG_SHAPE),
                                                 class_mode='binary')


#훈련 모델 정의
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(150, 150, 1)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    
    tf.keras.layers.Dropout(0.5), # 드롭 아웃 레이어로 들어오는 값의 50%가 0으로 재설정
    #  이를 통해 모델의 복원력이 향상
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(2, activation='softmax') #확률 분포
    #tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary() #확인


# 모델 훈련
epochs=100
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=int(np.ceil(train_size / float(BATCH_SIZE))),
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=int(np.ceil(val_size / float(BATCH_SIZE)))
)


# 그래프로 정확도 확인
acc = history.history['accuracy'] 
loss = history.history['loss']

epochs_range = range(epochs)

plt.figure(figsize=(8,8))
plt.subplot(1,2,1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1,2,2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.savefig('./foo.png')
plt.show()

model.save('model.h5')