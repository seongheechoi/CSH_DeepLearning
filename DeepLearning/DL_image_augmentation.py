'''
# 데이터 다운로드
import gdown
url = 'https://drive.google.com/uc?id=1nBE3N2cXQGwD8JaD0JZ2LmFD-n3D5hVU'
fname = 'cats_and_dogs_small.zip'
gdown.download(url, fname, quiet=False)
'''

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import numpy as np
import os
os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
np.random.seed(1)
tf.random.set_seed(1)

learning_rate = 0.001
dropout_rate = 0.5
N_EPOCHS = 50
N_BATCH = 20

# 모델 정의
def create_model():
    model = keras.Sequential()
    # Feature Extraction(특성 추출 - 이미지의 특징을 추출하는 layer)
    # Conv2D - keras_size: 3, padding = 'SAME, stride = 1(기본값)
    # MaxPool2D - pool_size : 2(기본), padding = 'SAME
    model.add(layers.Conv2D(64, kernel_size=3, padding='SAME', activation='relu', input_shape=(150,150,3)))
    model.add(layers.MaxPool2D(padding = 'SAME'))
    model.add(layers.Conv2D(64, kernel_size=3, padding='SAME', activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D(padding='SAME'))
    model.add(layers.Conv2D(128, kernel_size=3, padding='SAME', activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D(padding='SAME'))
    model.add(layers.Conv2D(128, kernel_size=3, padding='SAME', activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D(padding='SAME'))

    # classification layer (분류 - Dense)
    model.add(layers.Flatten())
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    return model

# 데이터 전처리
'''
* 사진 파일 읽기
* JPEG content를 RGB 픽셀 값으로 디코딩
* floating point tensor 형태로 변환
0-255 사이의 값을 가지는 픽셀 값을 0-1 사이 값으로 변환
keras.preprocessing.image.ImageDataGenerator 를 사용하여 자동으로 입력 가능한 형태로 변환 가능
'''
# train, validation, test 이미지가 들어있는 폴더 경로를 지정
train_dir = './cats_and_dogs_small/train'
validation_dir = './cats_and_dogs_small/validation'
test_dir = './cats_and_dogs_small/test'

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# ImageDataGenerator 생성 - 각 픽셀을 0~1 사이로 범위 조정(scaling)
train_datagen = ImageDataGenerator(rescale=1/255)
test_datagen = ImageDataGenerator(rescale=1/255)

# 이미지와 연결 -> generator 생성
# train set
train_generator = train_datagen.flow_from_directory(train_dir, target_size=(150,150), class_mode='binary', batch_size=N_BATCH)
# validation set
validation_generator = test_datagen.flow_from_directory(validation_dir, target_size=(150,150), class_mode='binary', batch_size=N_BATCH)
# test set
test_generator = test_datagen.flow_from_directory(test_dir, target_size=(150,150), class_mode='binary', batch_size=N_BATCH)

# label 클래스 확인
train_generator.class_indices #{'cats': 0, 'dogs': 1} 디렉토리 내 폴더 정보 확인
# step 수 (에폭당 몇번 weight 업데이트 할 것인지)
len(train_generator), len(validation_generator), len(test_generator) # (100, 50, 50)
# 배치개수만큼 이미지 조회
batch = train_generator.next() #2
len(batch), type(batch) #(2, tuple)
type(batch[0]), batch[0].shape #(numpy.ndarray, (20, 150, 150, 3))

'''
# 이미지 확인
plt.figure(figsize=(10,10))
for i in range(4):
    plt.subplot(2,2,i+1)
    plt.imshow(batch[0][i])
plt.show()
'''

# 모델 생성
model = create_model()
model.compile(optimizer=keras.optimizers.Adam(learning_rate), loss='binary_crossentropy', metrics=['accuracy'])

# 학습
history = model.fit(train_generator,
                    epochs=N_EPOCHS,
                    steps_per_epoch=len(train_generator),
                    validation_data=validation_generator,
                    validation_steps=len(validation_generator)
                    )

