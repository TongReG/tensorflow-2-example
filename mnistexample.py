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

fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# let's display the first 25 images from the training set and display the class
# name below each image.
r, c = 7, 7
plt.figure(figsize=(r,c))
for i in range(r * c):
    plt.subplot(r,c,i + 1)
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
#transforms the format of the images from a two-dimensional array (of 28 by 28
#pixels)
#to a one-dimensional array (of 28 * 28 = 784 pixels).
#Think of this layer as unstacking rows of pixels in the image and lining them
#up.
#This layer has no parameters to learn; it only reformats the data.

#After the pixels are flattened, the network consists of a sequence of two
#tf.keras.layers.Dense layers.
#These are densely connected, or fully connected, neural layers.
#The first Dense layer has 128 nodes (or neurons).
#The second (and last) layer returns a logits array with length of 10.
#Each node contains a score that indicates the current image belongs to one of
#the 10 classes.
model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#训练并验证模型：
#model.fit(x_train, y_train, epochs=10)
#test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

#print('\nMNIST Test accuracy:', test_acc)
model.fit(train_images, train_labels, epochs=15)
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

print('\nMNIST FASHION Test accuracy:', test_acc)

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)

#Graph this to look at the full set of 10 class predictions.
def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array, true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100 * np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array, true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')

# Plot the first X test images, their predicted labels, and the true labels.
# Color correct predictions in blue and incorrect predictions in red.
for j in range(10):
    num_rows = 6
    num_cols = 6
    num_images = num_rows * num_cols
    plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
    for i in range(num_images):
      plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
      plot_image(i + j * num_images, predictions[i + j * num_images], test_labels, test_images)
      plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
      plot_value_array(i + j * num_images, predictions[i + j * num_images], test_labels)
    plt.tight_layout()
    plt.show()#show(False)


# Grab an image from the test dataset.
img = test_images[1]

print(img.shape)

# Add the image to a batch where it's the only member.
img = (np.expand_dims(img,0))

print(img.shape)

predictions_single = probability_model.predict(img)

print(predictions_single)

plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)

np.argmax(predictions_single[0])






