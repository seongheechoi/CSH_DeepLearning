# from keras.datasets import mnist
#
# (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
#
# print(train_images.ndim)
# print(train_images.shape)
# print(train_images.dtype)
#
# digit = train_images[4]
#
# import matplotlib.pyplot as plt
#
# my_slice = train_images[10:100]
# print(my_slice.shape)
#
# # my_slice2 = train_images[:, 14:, 14:]
# # print(my_slice2.shape)
#
# def naive_relu(x):
#     assert len(x.shape) == 2
#     x = x.copy()      #입력텐서를 바꾸지 않도록 복사
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             x[i, j] = max(x[i, j], 0)
#     return x
#
# def naive_add(x, y):
#     assert len(x.shape) == 2
#     assert x.shape == y.shape
#
#     x = x.copy()
#     for i in range(x.shape[0]):
#         for j in range(x.shape[1]):
#             x[i, j] += y[i, j]
#     return x

import numpy as np
import os
import time

os.environ['KERAS_BACKEND'] = 'plaidml.keras.backend'
import keras
import keras.applications as kapp
from keras.datasets import cifar10

(x_train, y_train_cats), (x_test, y_test_cats) = cifar10.load_data()
batch_size = 8
x_train = x_train[:batch_size]
x_train = np.repeat(np.repeat(x_train, 7, axis=1), 7, axis=2)
print('1')
model = kapp.VGG19()
model.compile(optimizer='sgd', loss='categorical_crossentropy',
              metrics=['accuracy'])

print("Running initial batch (compiling tile program)")
y = model.predict(x=x_train, batch_size=batch_size)

# Now start the clock and run 10 batches
print("Timing inference...")
start = time.time()
for i in range(10):
    y = model.predict(x=x_train, batch_size=batch_size)
    print(i)
print("Ran in {} seconds".format(time.time() - start))