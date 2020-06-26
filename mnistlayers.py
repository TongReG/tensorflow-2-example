# MNIST Overfit
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用cpu，不注释就是使用gpu
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import pathlib
import shutil
import tempfile
#pip install -q git+https://github.com/tensorflow/docs
import tensorflow_docs as tfdocs
import tensorflow_docs.modeling
import tensorflow_docs.plots

logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
shutil.rmtree(logdir, ignore_errors=True)

#载入并准备好 MNIST 数据集。将样本从整数转换为浮点数：
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# let's display the first 25 images from the training set and display the class
# name below each image.
train_images = train_images / 255.0
test_images = test_images / 255.0

N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

#Next include callbacks.EarlyStopping to avoid long and unnecessary training
#times.
#Note that this callback is set to monitor the val_binary_crossentropy, not the
#val_loss.
#This difference will be important later.
def get_callbacks(name):
  return [tfdocs.modeling.EpochDots(),
    tf.keras.callbacks.EarlyStopping(monitor='val_binary_crossentropy', patience=200),
    tf.keras.callbacks.TensorBoard(logdir / name),]


def compile_and_fit(model, name, optimizer='adam', max_epochs=20):
  if optimizer is None:
    optimizer = get_optimizer()
  model.compile(optimizer=optimizer,
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=[tf.keras.losses.BinaryCrossentropy(from_logits=True, name='binary_crossentropy'),
                  'accuracy'])

  model.summary()

  history = model.fit(train_images, train_labels,
    epochs=max_epochs,
    validation_data=(test_images, test_labels),
    callbacks=get_callbacks(name),
    verbose=2)

  #history = model.fit(train_images, train_labels,
  #  steps_per_epoch = STEPS_PER_EPOCH,
  #  epochs=max_epochs,
  #  validation_data=validate_ds,
  #  callbacks=get_callbacks(name),
  #  verbose=2)
  return history


size_histories = {}
regularizer_histories = {}

shutil.rmtree(logdir / 'regularizers/Normal', ignore_errors=True)
#shutil.copytree(logdir / 'sizes/Normal', logdir / 'regularizers/Normal')


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
  tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')])


large_model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')])


size_histories['Normal'] = compile_and_fit(model, 'sizes/Normal')
regularizer_histories['Normal'] = size_histories['Normal']
regularizer_histories['large'] = compile_and_fit(large_model, "regularizers/large")

#训练并验证模型：
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
large_loss, large_acc = large_model.evaluate(test_images, test_labels, verbose=2)

print('\nMNIST FASHION Normal accuracy:', test_acc)
print('\nMNIST FASHION Large accuracy:', large_acc)

plotter = tfdocs.plots.HistoryPlotter(metric = 'binary_crossentropy', smoothing_std=10)

plotter.plot(size_histories)
a = plt.xscale('log')
plt.xlim([5, max(plt.xlim())])
plt.ylim([0, 0.9])
plt.xlabel("Epochs [Log Scale]")

plotter.plot(regularizer_histories)
plt.ylim([0.2, 0.9])


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




