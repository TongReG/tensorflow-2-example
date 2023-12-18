# 基础4
# 使用 KerasAPI 创建 MobileNetv2，训练CIFAR100数据集，并自动保存恢复结果

import os
import tensorflow as tf
from matplotlib import pyplot


# 对tensorflow环境进行设置
def configTFEnviron():
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行用于屏蔽GPU设备，以便使用CPU进行训练
    # tf.random.set_seed(2345)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    # tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
    # 启用设备放置日志记录将导致打印任何张量分配或操作
    # tf.debugging.set_log_device_placement(True)

# 预处理
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

if __name__ == '__main__':

    configTFEnviron()

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar100.load_data()
    train_label_s = tf.squeeze(train_labels, axis=1)
    test_label_s = tf.squeeze(test_labels, axis=1)

    # batch就是将多个元素组合成batch
    # shuffle的功能为打乱dataset中的元素，参数buffersize表示打乱时使用的buffer的大小
    batchsize = 128
    train_data = tf.data.Dataset.from_tensor_slices((train_images, train_label_s))
    train_data = train_data.shuffle(buffer_size=1024).map(preprocess).batch(batchsize)
    test_data = tf.data.Dataset.from_tensor_slices((test_images, test_label_s))
    test_data = test_data.map(preprocess).batch(batchsize)

    # 这一部分打印train_data的信息
    sample = next(iter(train_data))
    print('BatchSize =', batchsize, '\n')
    print('sample:', sample[0].shape, sample[1].shape,
          tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))

    # 通过Keras的API，创建MobileNetV2模型：https://keras.io/zh/applications/#mobilenetv2
    std_mnetv2 = tf.keras.applications.mobilenet_v2.MobileNetV2(weights=None,
                                                                pooling='max',
                                                                input_shape=(32, 32, 3),
                                                                alpha=0.5,
                                                                include_top=True,
                                                                classes=100,
                                                                classifier_activation='softmax')


    if not os.path.exists("mnetv2/"):
        os.makedirs("mnetv2/")
        if not os.path.exists("mnetv2/weights"):
            os.makedirs("mnetv2/weights")

    checkpoint_mnetv2path = "mnetv2/weights"
    # 使用Keras创建一个检查点回调，参考：https://blog.csdn.net/zengNLP/article/details/94589469
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_mnetv2path,
                                                     verbose=0,
                                                     save_best_only=False,
                                                     save_weights_only=False,
                                                     save_freq='epoch',
                                                     mode='auto',
                                                     patience=2)
    # https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks/CSVLogger
    csvlog = tf.keras.callbacks.CSVLogger("mnetv2/traincsv.log", separator=',', append=True)


    try:
        mnetv2_reload = tf.keras.models.load_model(checkpoint_mnetv2path)
        print('\nLatest PB data Load Success. Restore from PB model Now.\n')
        mnetv2_reload.summary()
        reloadstate = True
    except Exception as exc:
        print("\nException catched as : %s" % exc)
        print('\nLatest PB data Load Failed! Restart training steps.\n')
        reloadstate = False
        ckpt_num = 0

    # 绘制训练结果
    EpochArr = []
    AccArr, valAccArr = [], []
    tlossArr, valossArr = [], []
    if os.path.exists("mnetv2/traincsv.log"):
        graduate = []
        logf = open("mnetv2/traincsv.log", "r", encoding='utf-8')
        firstline = True
        cnt = 0
        for lines in logf.readlines():  # 遍历每一行
            ckpt = lines.split(',')
            if not firstline:
                EpochArr.append(int(ckpt[0]))
                AccArr.append(float(ckpt[1]))
                tlossArr.append(float(ckpt[2]))
                valAccArr.append(float(ckpt[3]))
                valossArr.append(float(ckpt[4]))
            firstline = False
            cnt = cnt + 1
        logf.close()
        graduate = []
        deGraduate = 5
        # 计算y的刻度值
        for i in range(len(tlossArr)):
            if i * deGraduate < max(tlossArr) + deGraduate:
                graduate.append(i * deGraduate)
        ckpt_num = cnt  # ckpt_num = max(EpochArr)
        drawLine(tlossArr, valossArr, "Epoches", "(val)Loss",
                 "Loss function curve", graduate)
        drawLine(AccArr, valAccArr, "Epoches", "(val)Accuracy",
                 "Accuracy function curve", [0, 0.25, 0.5, 0.75, 1])
    else:
        ckpt_num = 0

    epoch_num = 150

    def scheduler(epoch):
        lr_scheduler = []
        lr = 1e-2
        lr_scheduler.append(lr)
        reduce_grad = 0.9
        for i in range(0, epoch_num):
            if i >= 1:
                last_lr = lr_scheduler[i - 1] * reduce_grad
                lr_scheduler.append(last_lr)
        return lr_scheduler[epoch + ckpt_num]

    change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    # 经测试，建议的学习率最好小于1e-3
    opt = tf.keras.optimizers.Adam(lr=1e-4)
    earlystop = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy', patience=30, verbose=1, mode='auto')
    std_mnetv2.compile(loss='sparse_categorical_crossentropy',
                       optimizer=opt, metrics=['accuracy'])
    if reloadstate == False:
        std_mnetv2.fit(train_images, train_labels,
                       epochs=epoch_num,
                       batch_size=batchsize,
                       callbacks=[cp_callback, csvlog, change_lr, earlystop],
                       validation_data=(test_images, test_labels))
        large_loss, large_acc = std_mnetv2.evaluate(x=test_images, y=test_labels, verbose=0)
    else:
        std_mnetv2.fit(train_images, train_labels,
                       epochs=epoch_num - ckpt_num,
                       batch_size=batchsize,
                       callbacks=[cp_callback, csvlog, earlystop],
                       validation_data=(test_images, test_labels))
        large_loss, large_acc = std_mnetv2.evaluate(x=test_images, y=test_labels, verbose=0)

    # 验证结果
    test_loss, test_acc = std_mnetv2.evaluate(x=test_images, y=test_labels, verbose=0)
    print('\nCIFAR100 MobileNetV2 val_loss/accurary:', large_loss, large_acc)

    if reloadstate == False:
        std_mnetv2.save('cifar100_mnetv2.h5')