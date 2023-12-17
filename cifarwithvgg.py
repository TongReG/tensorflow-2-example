# 基础3.1
# 通过Keras搭建VGG13网络，训练CIFAR1O/100数据集，并自动保存恢复结果

import os
import re
import time
import tensorflow as tf
from matplotlib import pyplot

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉可以调用GPU，不注释时使用CPU
# tf.random.set_seed(2345)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(device=physical_devices[0],enable=True)
# 启用设备放置日志记录将导致打印任何张量分配或操作
# tf.debugging.set_log_device_placement(True)
try:  # 如果模型存在则直接预加载，不再训练
    vgg13_net = tf.keras.models.load_model('cifar10_vgg13.h5')
    fc_net = tf.keras.models.load_model('cifar10_vgg13fc.h5')
    VGG13STATE = True
    print('\nVGG13 Model Load Successful.\n')
except Exception as e:
    print("\nException catched as : %s" % e)
    print('\nVGG13 Model Load failed ! Restart training steps.\n')
    VGG13STATE = False

# 预处理数据
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    y = tf.cast(y, dtype=tf.int32)
    return x, y

# 训练后生成图表
def drawLine(arr, arr2, xName, yName, title, graduate):
    x = [x + 1 for x in range(len(arr))]  # 横坐标 采用列表表达式
    y, y2 = arr, arr2                   # 纵坐标
    pyplot.plot(x, y, label="train")    # 生成折线图
    pyplot.plot(x, y2, label="val")
    pyplot.xlabel(xName)                # 设置横坐标说明
    pyplot.ylabel(yName)                # 设置纵坐标说明
    pyplot.legend()
    pyplot.title(title)                 # 添加标题
    pyplot.yticks(graduate)             # 设置纵坐标刻度
    pyplot.grid(True)                   # 显示网格
    pyplot.show()                       # 显示图表


batchsize = 128

(train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
# (train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
train_label_s = tf.squeeze(train_labels, axis=1)
test_label_s = tf.squeeze(test_labels, axis=1)

# batch就是将多个元素组合成batch
# shuffle的功能为打乱dataset中的元素，参数buffersize表示打乱时使用的buffer的大小
train_data = tf.data.Dataset.from_tensor_slices((train_images, train_label_s))
train_data = train_data.shuffle(buffer_size=1024).map(preprocess).batch(batchsize)

# 通过 tf.keras.utils.to_categorical 将数据转换为one hot格式
#train_labels = tf.keras.utils.to_categorical(train_labels, 10)
#test_labels = tf.keras.utils.to_categorical(test_labels, 10)
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_label_s))
test_data = test_data.map(preprocess).batch(batchsize)

# 这一部分打印train_data的信息
sample = next(iter(train_data))
print('VGG13 BatchSize =', batchsize, '\n')
print('sample:', sample[0].shape, sample[1].shape,
      tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

# 构建vgg13的结构
vgg13_layers = [tf.keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                tf.keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

                tf.keras.layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                tf.keras.layers.Conv2D(128, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

                tf.keras.layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                tf.keras.layers.Conv2D(256, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

                tf.keras.layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                tf.keras.layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),

                tf.keras.layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                tf.keras.layers.Conv2D(512, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
                tf.keras.layers.MaxPool2D(pool_size=[2, 2], strides=2, padding="same"),]

vgg13_net = tf.keras.Sequential(vgg13_layers)
# 全连接层
fc_net = tf.keras.Sequential([tf.keras.layers.Dense(4096, activation=tf.nn.relu),
                              tf.keras.layers.Dense(
                                  4096, activation=tf.nn.relu),
                              tf.keras.layers.Dense(10, activation='softmax')])

vgg13_net.build(input_shape=[None, 32, 32, 3])
fc_net.build(input_shape=[None, 512])
optimizer = tf.keras.optimizers.Adam(lr=1e-4)

vgg13_net.summary()
fc_net.summary()

# 创建检查点，指定最多保存3个检查点
checkpoint_vgg13 = tf.train.Checkpoint(model=vgg13_net, myOptimizer=optimizer)
checkpoint_vgg13fc = tf.train.Checkpoint(model=fc_net, myOptimizer=optimizer)
checkpoint_vgg13dir = os.path.dirname("vgg13/")
checkpoint_vgg13fcdir = os.path.dirname("vgg13fc/")
if not os.path.exists(checkpoint_vgg13dir):
    os.makedirs(checkpoint_vgg13dir)
    os.makedirs(checkpoint_vgg13fcdir)
ckptmngr_vgg13 = tf.train.CheckpointManager(checkpoint_vgg13, directory=checkpoint_vgg13dir, max_to_keep=3)
ckptmngr_vgg13fc = tf.train.CheckpointManager(checkpoint_vgg13fc, directory=checkpoint_vgg13fcdir, max_to_keep=3)

vgg13_latestpoint = tf.train.latest_checkpoint(checkpoint_vgg13dir)
print('\nVGG13 Latest traindata:', vgg13_latestpoint, '\n')


# 尝试从检查点恢复
try:
    ckpt_num = re.findall(r"\d+\.?\d*", vgg13_latestpoint)
    print('\nCKPT_NUM:', ckpt_num, '\n')
    ckpt_num = int(ckpt_num[1]) + 1
    checkpoint_vgg13.restore(ckptmngr_vgg13.latest_checkpoint)
    checkpoint_vgg13fc.restore(ckptmngr_vgg13fc.latest_checkpoint)
    print('\nVGG13 checkpoint Load Successful.\n')
    vgg13_restorestate = True
except Exception as ex:
    print("\nException catched as : %s" % ex)
    print('\nVGG13 checkpoint Load Failed.\n')
    vgg13_restorestate = False
    ckpt_num = 0


# tf.trainable_variables()函数可以也仅可以查看可训练的变量
variables = vgg13_net.trainable_variables + fc_net.trainable_variables
flag = 1
epoch_num = 30
for epoch in range(ckpt_num, epoch_num):
    elapsed_epoch = 0.0
    # 每个Epoch都要输入50000张图片, 则步数 steps = 50000 ÷ batchsize
    for step, (x, y) in enumerate(train_data):
        if flag == 1:
            start = time.perf_counter()
            flag = 0
        with tf.GradientTape() as tape:
            out = vgg13_net(x)
            out = tf.reshape(out, [-1, 512])
            logits = fc_net(out)
            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, variables)  # 计算梯度
        optimizer.apply_gradients(zip(grads, variables))  # 更新梯度
        if step % 8 == 0:
            elapsed = (time.perf_counter() - start)
            elapsed_epoch += elapsed
            flag = 1
            print('Epoch:', epoch, 'Step:', step, 'datas:', step * batchsize, 'loss:', '%.4f' % float(loss))
            print('Time:', '%.4f' %elapsed, 'EpochTime:', '%.4f' % elapsed_epoch)

    total_num = 0
    total_correct = 0
    for x, y in test_data:
        out = vgg13_net(x)
        out = tf.reshape(out, [-1, 512])
        logits = fc_net(out)
        pred = tf.argmax(logits, axis=1)  # axis=1，返回每一行最大元素所在
        pred = tf.cast(pred, dtype=tf.int32)
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_num += x.shape[0]
        total_correct += int(correct)
    acc = total_correct / total_num
    print('Epoch_End :', epoch, 'Accurary :', acc, 'Correct_num :', total_correct)

    ckptmngr_vgg13.save(checkpoint_number=epoch)
    ckptmngr_vgg13fc.save(checkpoint_number=epoch + epoch_num)
    print('Checkpoint Saved by Manager.\n')

    # try:
    #    vgg13_net.save_weights(os.path.join(checkpoint_dir,'vgg13_cp_{epoches:04d}'.format(epoches
    #    = epoch)))
    #    fc_net.save_weights(os.path.join(checkpoint_dir,'vgg13fc_cp_{epoches:04d}'.format(epoches
    #    = epoch)))
    #    print('Epoch weight Saved')
    # except: print('Epoch weight Save FAILED!!!!')

# 验证训练结果
test_loss, test_acc = vgg13_net.evaluate(x=test_images, y=test_labels, verbose=0)
print('\nCIFAR10 VGG13 val_loss/accurary:', test_loss, test_acc)
# print('\nCIFAR100 VGG13 val_loss/accurary:' , test_loss, test_acc)

# 保存模型
if not vgg13_restorestate:
    if not VGG13STATE:
        vgg13_net.save('cifar10_vgg13.h5')
        fc_net.save('cifar10_vgg13fc.h5')
