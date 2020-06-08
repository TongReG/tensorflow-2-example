#基础2 MNIST
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用cpu，不注释就是使用gpu
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

#载入并准备好 MNIST 数据集。将样本从整数转换为浮点数：
mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

#fashion_mnist = keras.datasets.fashion_mnist
#(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# let's display the first 25 images from the training set and display the class name below each image.
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

train_images = train_images / 255.0
test_images = test_images / 255.0

#将模型的各层堆叠起来，以搭建 tf.keras.Sequential 模型。为训练选择优化器和损失函数

#The first layer in this network, tf.keras.layers.Flatten
#transforms the format of the images from a two-dimensional array (of 28 by 28 pixels) 
#to a one-dimensional array (of 28 * 28 = 784 pixels). 
#Think of this layer as unstacking rows of pixels in the image and lining them up. 
#This layer has no parameters to learn; it only reformats the data.

#After the pixels are flattened, the network consists of a sequence of two tf.keras.layers.Dense layers.
#These are densely connected, or fully connected, neural layers. 
#The first Dense layer has 128 nodes (or neurons). 
#The second (and last) layer returns a logits array with length of 10. 
#Each node contains a score that indicates the current image belongs to one of the 10 classes.
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#训练并验证模型：
model.fit(x_train, y_train, epochs=10)
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)

print('\nTest accuracy:', test_acc)








