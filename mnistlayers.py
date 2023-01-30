# 基础2-2 MNIST 过拟合实例
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import pathlib
import shutil
import tempfile
# 使用pip install -q git+https://github.com/tensorflow/docs 安装doc内容
import tensorflow_docs as tfdocs
import tensorflow_docs.plots


def modelrestore():
    try:  # 如果模型存在则直接预加载，不再训练
        model = tf.keras.models.load_model('fashion_mnist_normal.h5')
        large_model = tf.keras.models.load_model('fashion_mnist_large.h5')
        MODELSTATE = True
    except:
        print('\nModel Load failed ! Restart training steps...\n')
        MODELSTATE = False
    return MODELSTATE


def modelsave(model, is_large):
    # 保存模型的权重和偏置
    try:
        if is_large:
            model.save('fashion_mnist_large.h5')
        else:
            model.save('fashion_mnist_normal.h5')  # creates a HDF5 file
    except:
        print('\nModel Save failed !')


def get_callbacks(name):
    # tfdocs.modeling.EpochDots()
    # Include callbacks.EarlyStopping to avoid long and unnecessary training times.
    # Note that this callback is set to monitor the val_binary_crossentropy, not the
    # val_loss. This difference will be important later.

    # monitor: 被监测的数据。
    # min_delta: 在被监测的数据中被认为是提升的最小变化，例如，小于 min_delta 的绝对变化会被认为没有提升。
    # patience: 没有进步的训练轮数，在这之后训练就会被停止。
    # verbose: 详细信息模式。
    return [tf.keras.callbacks.EarlyStopping(monitor='val_sparse_categorical_crossentropy', patience=25),
            tf.keras.callbacks.TensorBoard(logdir / name)]


def compile_and_fit(model, name, optimizer='adam', max_epochs=120):
    # 搭建模型并开始训练，返回history对象
    if optimizer is None:
        optimizer = get_optimizer()
    model.compile(optimizer=optimizer,
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(
                      from_logits=True, name='sparse_categorical_crossentropy'),
                  metrics=[tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, name='sparse_categorical_crossentropy'),
                           'accuracy'])

    model.summary()

    # steps_per_epoch 代表每个epoch被分割的步数，减少就意味着一次输入多个数据，加快训练速度。
    history = model.fit(train_images, train_labels,
                        epochs=max_epochs,
                        steps_per_epoch=468,
                        validation_data=(test_images, test_labels),
                        callbacks=get_callbacks(name),
                        verbose=2)
    return history


# Graph this to look at the full set of 10 class predictions.
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


def show_predict_result(predictions, rows, cols, pages):
    # Plot the first row*col test images, their predicted labels, and the true labels.
    # 正确的predictions标记为蓝，错误的predictions标记为红。
    for j in range(pages):
        num_rows = rows
        num_cols = cols
        num_images = num_rows * num_cols
        plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
        for i in range(num_images):
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
            plot_image(i + j * num_images,
                       predictions[i + j * num_images], test_labels, test_images)
            plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
            plot_value_array(i + j * num_images,
                             predictions[i + j * num_images], test_labels)
        plt.tight_layout()
        plt.show()  # show(False)


def std_model():
    # The first layer in this network, tf.keras.layers.Flatten
    # transforms the format of the images from a two-dimensional array (of 28 by 28
    # pixels)
    # to a one-dimensional array (of 28 * 28 = 784 pixels).
    # Think of this layer as unstacking rows of pixels in the image and lining them
    # up.
    # This layer has no parameters to learn; it only reformats the data.

    # After the pixels are flattened, the network consists of a sequence of two
    # tf.keras.layers.Dense layers.
    # These are densely connected, or fully connected, neural layers.
    # The first Dense layer has 128 nodes (or neurons).
    # The second (and last) layer returns a logits array with length of 10.
    # Each node contains a score that indicates the current image belongs to one of
    # the 10 classes.
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                        tf.keras.layers.Dense(128, activation='relu',
                                                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.Dense(128, activation='relu',
                                                              kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.Dense(10, activation='softmax')])
    return model


def huge_model():
    large_model = tf.keras.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                       tf.keras.layers.Dense(256, activation='relu',
                                                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                                       tf.keras.layers.Dropout(0.3),
                                       tf.keras.layers.Dense(256, activation='relu',
                                                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                                       tf.keras.layers.Dropout(0.3),
                                       tf.keras.layers.Dense(256, activation='relu',
                                                             kernel_regularizer=tf.keras.regularizers.l2(0.001)),
                                       tf.keras.layers.Dropout(0.3),
                                       tf.keras.layers.Dense(10, activation='softmax')])
    return large_model


# 一些全局变量
N_VALIDATION = int(1e3)
N_TRAIN = int(1e4)
BUFFER_SIZE = int(1e4)
BATCH_SIZE = 500
STEPS_PER_EPOCH = N_TRAIN // BATCH_SIZE

MODELSTATE = False


if __name__ == "__main__":
    # 设置log路径
    logdir = pathlib.Path(tempfile.mkdtemp()) / "tensorboard_logs"
    shutil.rmtree(logdir, ignore_errors=True)

    # 载入并准备好 MNIST FASHION 数据集。将样本从整数转换为浮点数：
    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    size_histories = {}
    regularizer_histories = {}

    shutil.rmtree(logdir / 'regularizers/Normal', ignore_errors=True)
    #shutil.copytree(logdir / 'sizes/Normal', logdir / 'regularizers/Normal')
    # 如果模型已存档好，则直接加载
    MODELSTATE = modelrestore()

    # 将模型的各层堆叠起来，以搭建 tf.keras.Sequential 模型。为训练选择优化器和损失函数
    if not MODELSTATE:
        # 选择训练大模型或者小模型
        while True:
            choice = input("Training large model? (Y/N/Q): ")
            if choice.upper() == "N":
                model = std_model()
                # 训练并验证模型
                size_histories['Normal'] = compile_and_fit(
                    model, 'sizes/Normal')
                regularizer_histories['Normal'] = size_histories['Normal']
                test_loss, test_sparse_categorical_crossentropy, test_acc = model.evaluate(
                    x=test_images, y=test_labels, verbose=0, callbacks=get_callbacks('sizes/Normal'))
                print('\nMNIST FASHION Normal sparse categorical crossentropy: ',
                      test_sparse_categorical_crossentropy)
                print('\nMNIST FASHION Normal val_loss/accurary: ',
                      test_loss, test_acc)
                # 保存模型
                modelsave(model=model, is_large=False)
                break
            elif choice.upper() == "Y":
                large_model = huge_model()
                # 训练并验证模型
                regularizer_histories['large'] = compile_and_fit(
                    large_model, "regularizers/large")
                large_loss, large_sparse_categorical_crossentropy, large_acc = large_model.evaluate(
                    x=test_images, y=test_labels, verbose=0, callbacks=get_callbacks("regularizers/large"))
                print('\nMNIST FASHION Large sparse categorical crossentropy: ',
                      large_sparse_categorical_crossentropy)
                print('\nMNIST FASHION Large val_loss/accurary: ',
                      large_loss, large_acc)
                # 保存模型
                modelsave(model=large_model, is_large=True)
                break
            elif choice.upper() == "Q":
                exit()
            else:
                print("Please input Y or N...")
                print("Input Q to Exit...")

    plotter = tfdocs.plots.HistoryPlotter(
        metric='sparse_categorical_crossentropy', smoothing_std=5)

    plotter.plot(size_histories)
    a = plt.xscale('log')
    plt.xlim([0, max(plt.xlim())])
    plt.ylim([0, 2])
    plt.xlabel("Epochs [Log Scale]")
    plt.ylabel("")
    plt.show()

    plotter.plot(regularizer_histories)
    plt.ylim([0, 2])

    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    predictions = probability_model.predict(test_images)

    probability_largemodel = tf.keras.Sequential([large_model,
                                                  tf.keras.layers.Softmax()])
    predictions_large = probability_largemodel.predict(test_images)
