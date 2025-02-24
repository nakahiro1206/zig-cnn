import numpy as np
# import mnist
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from keras.utils import to_categorical
from keras.optimizers import SGD
import time

# train_images = mnist.train_images()
# train_labels = mnist.train_labels()
# test_images = mnist.test_images()
# test_labels = mnist.test_labels()

def load(f):
    return np.load(f)['arr_0']

# Load the data
train_images = load('../../rois-codh/kmnist/kmnist-train-imgs.npz')
test_images = load('../../rois-codh/kmnist/kmnist-test-imgs.npz')
train_labels = load('../../rois-codh/kmnist/kmnist-train-labels.npz')
test_labels = load('../../rois-codh/kmnist/kmnist-test-labels.npz')

train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

train_images = np.expand_dims(train_images, axis=3)
test_images = np.expand_dims(test_images, axis=3)

model = Sequential([
  Conv2D(8, 3, input_shape=(28, 28, 1), use_bias=False),
  MaxPooling2D(pool_size=2),
  Flatten(),
  Dense(10, activation='softmax'),
])

model.compile(SGD(learning_rate=.005), loss='categorical_crossentropy', metrics=['accuracy'])

start = time.time()
model.fit(
  train_images,
  to_categorical(train_labels),
  batch_size=1,
  epochs=1,
  validation_data=(test_images, to_categorical(test_labels)),
)
print('Elapsed time:', time.time() - start)

'''
Epoch 1
46s 760us/step - loss: 0.2433 - acc: 0.9276 - val_loss: 0.1176 - val_acc: 0.9634
Epoch 2
46s 771us/step - loss: 0.1184 - acc: 0.9648 - val_loss: 0.0936 - val_acc: 0.9721
Epoch 3
48s 797us/step - loss: 0.0930 - acc: 0.9721 - val_loss: 0.0778 - val_acc: 0.9744
'''
