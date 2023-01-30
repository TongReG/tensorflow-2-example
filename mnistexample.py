# 基础2-1 MNIST基本训练
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np


def show_imgs(row, col, class_names):
    # 从数据集的开头显示row*col个图像，并且显示每张图的class name。
    plt.figure(figsize=(row, col))
    for i in range(row * col):
        plt.subplot(row, col, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(train_images[i], cmap=plt.cm.binary)
        plt.xlabel(class_names[train_labels[i]])
    plt.show()


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


def predict_all(model, imgs):
    # 用当前模型进行预测
    predictions = model.predict(imgs)
    # 显示预测结果
    show_predict_result(predictions, rows=6, cols=6, pages=4)


def predict_one_img(img, model, labels):
    # 进行单次预测
    # Grab an image from the test dataset.
    img = test_images[1]
    # Add the image to a batch where it's the only member.
    img = (np.expand_dims(img, 0))
    # print(img.shape)
    # 用模型开始预测
    predictions_single = model.predict(img)
    print(predictions_single)

    plot_value_array(1, predictions_single[0], labels)
    _ = plt.xticks(range(10), class_names, rotation=45)

    np.argmax(predictions_single[0])


def build_model():
    # The first layer in this network, tf.keras.layers.Flatten
    # transforms the format of the images from a two-dimensional array (of 28 by 28
    # pixels) to a one-dimensional array (of 28 * 28 = 784 pixels).
    # Think of this layer as unstacking rows of pixels in the image and lining them
    # up. This layer has no parameters to learn; it only reformats the data.

    # After the pixels are flattened, the network consists of a sequence of two
    # tf.keras.layers.Dense layers.
    # These are densely connected, or fully connected, neural layers.
    # The first Dense layer has 128 nodes (or neurons).
    # The second (and last) layer returns a logits array with length of 10.
    # Each node contains a score that indicates the current image belongs to one of the 10 classes.
    model = tf.keras.models.Sequential([tf.keras.layers.Flatten(input_shape=(28, 28)),
                                        tf.keras.layers.Dense(
                                            128, activation='relu'),
                                        tf.keras.layers.Dropout(0.2),
                                        tf.keras.layers.Dense(10, activation='softmax')])
    # 为训练选择优化器和损失函数
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


if __name__ == "__main__":
    # 载入并准备好 MNIST 数据集。将样本从整数转换为浮点数：
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    fashion_mnist = tf.keras.datasets.fashion_mnist
    (train_images, train_labels), (test_images,
                                   test_labels) = fashion_mnist.load_data()
    train_images = train_images / 255.0
    test_images = test_images / 255.0

    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                   'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
    # 显示图片
    show_imgs(7, 7, class_names=class_names)

    # 将模型的各层堆叠起来，以搭建 tf.keras.Sequential 模型。
    model = build_model()

    # 训练并验证模型：
    #model.fit(x_train, y_train, epochs=10)
    #test_loss, test_acc = model.evaluate(x_test, y_test, verbose=1)

    #print('\nMNIST Test accuracy:', test_acc)
    model.fit(train_images, train_labels, epochs=25)
    test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)

    print('\nMNIST FASHION Test accuracy:', test_acc)

    # 合并一个softmax层，用作预测模型
    probability_model = tf.keras.Sequential([model,
                                             tf.keras.layers.Softmax()])
    # 使用验证集的图片进行预测
    predict_all(model=probability_model, imgs=test_images)

    # 预测(识别)任意单个图像
    predict_one_img(
        test_images[1], model=probability_model, labels=test_labels)
