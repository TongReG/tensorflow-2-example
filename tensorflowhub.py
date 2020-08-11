#基础3 使用 Keras 和 Tensorflow Hub 对电影评论进行文本二分类
#使用来源于网络电影数据库（Internet Movie Database）的 IMDB 数据集（IMDB dataset），
#其包含 50,000 条影评文本。从该数据集切割出的 25,000 条评论用作训练，另外 25,000 条用作测试。
#训练集与测试集是平衡的（balanced），意味着它们包含相等数量的积极和消极评论。
#此笔记本（notebook）使用了 tf.keras，它是一个 Tensorflow 中用于构建和训练模型的高级API，
#此外还使用了 TensorFlow Hub，一个用于迁移学习的库和平台。
from __future__ import absolute_import, division, print_function, unicode_literals
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉可以调用GPU，不注释时使用CPU
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub
import tensorflow_datasets as tfds

print("Version: ", tf.__version__)
print("Eager mode: ", tf.executing_eagerly())
print("Hub version: ", hub.__version__)
print("GPU is", "available" if tf.config.experimental.list_physical_devices("GPU") else "NOT AVAILABLE")

# 将训练集按照 6:4 的比例进行切割，从而最终我们将得到
# 15,000 个训练样本, 10,000 个验证样本以及 25,000 个测试样本
train_validation_split = tfds.Split.TRAIN.subsplit([6, 4])

(train_data, validation_data), test_data = tfds.load(name="imdb_reviews", 
    split=(train_validation_split, tfds.Split.TEST),
    as_supervised=True)

#让我们花一点时间来了解数据的格式。每一个样本都是一个表示电影评论和相应标签的句子。
#该句子不以任何方式进行预处理。标签是一个值为 0 或 1 的整数，其中 0 代表消极评论，1 代表积极评论。
#我们来打印下前十个样本。
train_examples_batch, train_labels_batch = next(iter(train_data.batch(10)))
print(train_examples_batch, train_labels_batch)

#表示文本的一种方式是将句子转换为嵌入向量（embeddings vectors）。
#我们可以使用一个预先训练好的文本嵌入（text embedding）作为首层，这将具有三个优点：
#我们不必担心文本预处理
#我们可以从迁移学习中受益
#嵌入具有固定长度，更易于处理
#针对此示例我们将使用 TensorFlow Hub 中名为 google/tf2-preview/gnews-swivel-20dim/1
#的一种预训练文本嵌入（text embedding）模型 。
#为了达到本教程的目的还有其他三种预训练模型可供测试：
#google/tf2-preview/gnews-swivel-20dim-with-oov/1 ——类似
#google/tf2-preview/gnews-swivel-20dim/1，但 2.5%的词汇转换为未登录词桶（OOV
#buckets）。如果任务的词汇与模型的词汇没有完全重叠，这将会有所帮助。
#google/tf2-preview/nnlm-en-dim50/1 ——一个拥有约 1M 词汇量且维度为 50 的更大的模型。
#google/tf2-preview/nnlm-en-dim128/1 ——拥有约 1M 词汇量且维度为128的更大的模型。
embedding = "https://hub.tensorflow.google.cn/google/tf2-preview/gnews-swivel-20dim/1"
hub_layer = hub.KerasLayer(embedding, input_shape=[], 
                           dtype=tf.string, trainable=True)
hub_layer(train_examples_batch[:3])

model = tf.keras.Sequential()
model.add(hub_layer)
model.add(tf.keras.layers.Dense(16, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.summary()

#由于这是一个二分类问题且模型输出概率值（一个使用 sigmoid 激活函数的单一单元层）
#我们将使用 binary_crossentropy 损失函数。
#这不是损失函数的唯一选择，例如，您可以选择 mean_squared_error 。
#但是，一般来说 binary_crossentropy 更适合处理概率——它能够度量概率分布之间的“距离”，或者在我们的示例中，指的是度量
#ground-truth 分布与预测值之间的“距离”。
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

#以 512 个样本的 mini-batch 大小迭代 20 个 epoch 来训练模型。
#这是指对 x_train 和 y_train 张量中所有样本的的 20 次迭代。
#在训练过程中，监测来自验证集的 10,000 个样本上的损失值（loss）和准确率（accuracy）
history = model.fit(train_data.shuffle(10000).batch(512),
                    epochs=20,
                    validation_data=validation_data.batch(512),
                    verbose=1)

#损失值（loss）与准确率（accuracy）
results = model.evaluate(test_data.batch(512), verbose=2)
for name, value in zip(model.metrics_names, results):
    print("%s: %.3f" % (name, value))