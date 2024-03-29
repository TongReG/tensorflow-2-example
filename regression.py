# 基础4 Auto MPG 数据集回归 (regression) 问题
# 本 notebook 使用经典的 Auto MPG 数据集，构建了一个用来预测70年代末到80年代初汽车燃油效率的模型。
# 为了做到这一点，我们将为该模型提供许多那个时期的汽车描述。这个描述包含：气缸数，排量，马力以及重量。
from __future__ import absolute_import, division, print_function, unicode_literals
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


def download_DataSet():
    # 下载数据集
    dataset_path = tf.keras.utils.get_file(
        "auto-mpg.data",
        "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data")
    print(dataset_path)
    return dataset_path


def load_DataSet(dataset_path):
    # 使用 pandas 导入数据集
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path,
                              names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

    # 数据集中包括一些N/A值。为了保证例子的简单性，删除这些数据。
    dataset = raw_dataset.copy()
    dataset.tail().isna().sum()
    dataset = dataset.dropna()
    return dataset


def conv_OneHot(dataset):
    # "Origin" 列实际上代表分类，而不仅仅是一个数字。所以把它转换为独热码（one-hot）:
    origin = dataset.pop('Origin')
    dataset['USA'] = (origin == 1) * 1.0
    dataset['Europe'] = (origin == 2) * 1.0
    dataset['Japan'] = (origin == 3) * 1.0
    dataset.tail()
    return dataset


def show_Stats(dataset):
    # 也可以查看总体的数据统计:
    train_stats = dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()
    print(train_stats)


def normalize(x):
    # 使用不同的尺度和范围对特征归一化是好的实践。
    # 尽管模型可能在没有特征归一化的情况下收敛，但它会使得模型训练更加复杂，并会造成生成的模型依赖输入所使用的单位选择。
    train_stats = x.describe().transpose()
    return (x - train_stats['mean']) / train_stats['std']


def build_model():
    # 构建训练层
    model = tf.keras.Sequential([tf.keras.layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
                                 tf.keras.layers.Dense(64, activation='relu'),
                                 tf.keras.layers.Dense(1)])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


class PrintDot(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 50 == 0:
            print('50 Epoches Done')


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


def show_prediction(predictions, labels):
    # 绘制预测结果
    plt.scatter(labels, predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim([0, plt.xlim()[1]])
    plt.ylim([0, plt.ylim()[1]])
    _ = plt.plot([-100, 100], [-100, 100])
    plt.show()


if __name__ == "__main__":
    # 下载数据集，做预处理
    dataset_path = download_DataSet()
    dataset_initial = load_DataSet(dataset_path)
    dataset = conv_OneHot(dataset_initial)

    # 现在需要将数据集拆分为一个训练数据集和一个测试数据集，我们将使用测试数据集对模型进行评估。
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    show_Stats(train_dataset)
    show_Stats(test_dataset)

    # 将特征值从目标值或者"标签"中分离。 这个标签是你使用训练模型进行预测的值。
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    # 对数据进行归一化
    normed_train_data = normalize(train_dataset)
    normed_test_data = normalize(test_dataset)

    # 使用 .summary 方法来打印该模型的简单描述。
    model = build_model()
    model.summary()

    # 对模型进行1000个周期的训练，并在 history 对象中记录训练和验证的准确性。
    # 通过为每个完成的时期打印一个点来显示训练进度
    EPOCHS = 1000
    history = model.fit(normed_train_data, train_labels,
                        epochs=EPOCHS, validation_split=0.2, verbose=0,
                        callbacks=[PrintDot()])

    # 使用 history 对象中存储的统计信息可视化模型的训练进度。
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()
    # 把数据打印到图标中
    plot_history(history)

    # 图表显示，在约100个epochs之后，误差非但没有改进，反而出现恶化。

    # 让我们更新 model.fit 调用，当验证值没有提高时自动停止训练。
    model = build_model()
    # 我们将使用一个 EarlyStopping 回调来测试每个epoch的训练条件，
    # 如果经过一定数量的epochs后指标没有改进，则自动停止训练。
    # patience 即表示停止训练的阈值
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=10)

    history = model.fit(normed_train_data, train_labels,
                        epochs=EPOCHS, validation_split=0.2, verbose=0,
                        callbacks=[early_stop, PrintDot()])
    plot_history(history)

    # 让我们看看通过使用测试集来泛化模型的效果如何
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)
    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    # 最后，使用测试集中的数据预测 MPG 值:
    test_predictions = model.predict(normed_test_data).flatten()
    # 绘制结果
    show_prediction(test_predictions, test_labels)

    # 我们来看下误差分布。
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")


# 总结：
# 本笔记本 (notebook) 介绍了一些处理回归问题的技术。
# 均方误差（MSE）是用于回归问题的常见损失函数（分类问题中使用不同的损失函数）。
# 类似的，用于回归的评估指标与分类不同。 常见的回归指标是平均绝对误差（MAE）。
# 当数字输入数据特征的值存在不同范围时，每个特征应独立缩放到相同范围。
# 如果训练数据不多，一种方法是选择隐藏层较少的小网络，以避免过度拟合。
# 早期停止是一种防止过度拟合的有效技术。
