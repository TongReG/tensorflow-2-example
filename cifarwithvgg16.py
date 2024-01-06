# 基础3.2
# 通过Keras搭建VGG16网络，训练CIFAR1O/100数据集，并自动保存恢复结果

import os
import tensorflow as tf
from matplotlib import pyplot


# 对tensorflow环境进行设置
def configTFEnviron(useGPU):
    if not useGPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行用于屏蔽GPU设备，以便使用CPU进行训练
    # tf.random.set_seed(2345)
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    print(physical_devices)
    # tf.config.experimental.set_memory_growth(device=physical_devices[0], enable=True)
    # 启用设备放置日志记录将导致打印任何张量分配或操作
    # tf.debugging.set_log_device_placement(True)

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

# 设置可变的学习率。要用到的是model.fit中的callbacks参数，从参数名可以理解，我们需要写一个回调函数来实现学习率随训练轮数增加而减小。
# 这里我们让网络训练50个epoch，即epoch_num = 50
# 其中前20个采用0.01，中间20个采用0.001，最后10个采用0.0001
def scheduler(epoch):
    if epoch < epoch_num * 0.4:
        return learning_rate
    if epoch < epoch_num * 0.8:
        return learning_rate * 0.1
    return learning_rate * 0.01


if __name__ == '__main__':

    configTFEnviron(useGPU=True)

    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
    # (train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.cifar100.load_data()

    weight_decay = 5e-4
    dropout_rate = 0.5
    batch_size = 128
    learning_rate = 1e-2
    epoch_num = 40

    if not os.path.exists("vgg16/"):
        os.makedirs("vgg16/")
        os.makedirs("vgg16/weights")
        os.makedirs("vgg16/tensorboardlog")

    checkpoint_vgg16path = "vgg16/weights"
    #checkpoint_vgg16path = "vgg16/weights.{epoch:02d}"
    #checkpoint_vgg16dir = os.listdir("vgg16/")
    # for iters in checkpoint_vgg16dir:
    #    ckpt_num = re.findall(r"\d+\.?\d*",iters)
    #    ckpt_dir = iters
    #ckpt_num = int(ckpt_num[0])
    try:
        # os.path.join("vgg16/",ckpt_dir)
        vgg16_reload = tf.keras.models.load_model(checkpoint_vgg16path)
        print('\nVGG16 Latest PB data Load Success. Restore from PB model Now.\n')
        vgg16_reload.summary()
        vgg16_reloadstate = True
    except Exception as exc:
        print("\nException catched as : %s" % exc)
        print('\nVGG16 Latest PB data Load Failed! Restart training steps.\n')
        vgg16_reloadstate = False
        ckpt_num = 0

    # Keras方式创建一个检查点回调 https://blog.csdn.net/zengNLP/article/details/94589469
    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_vgg16path,
                                                     verbose=0,
                                                     save_best_only=False,
                                                     save_weights_only=False,
                                                     save_freq='epoch',
                                                     mode='auto',
                                                     patience=2)
    # https://tensorflow.google.cn/api_docs/python/tf/keras/callbacks/CSVLogger
    csvlog = tf.keras.callbacks.CSVLogger("vgg16/traincsv.log", 
                                          separator=',', append=True)
    # 添加tensorboard监控回调，以便在tensorboard上显示分析数据
    tensorboard = tf.keras.callbacks.TensorBoard(log_dir="vgg16/tensorboardlog")

    # 绘制训练结果
    EpochArr = []
    AccArr, valAccArr = [], []
    tlossArr, valossArr = [], []
    if os.path.exists("vgg16/traincsv.log"):
        graduate = []
        logf = open("vgg16/traincsv.log", "r", encoding='utf-8')
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

    # 构建vgg16的结构
    if not vgg16_reloadstate:
        vgg16_layers = [
                        # 输入层
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3), kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.MaxPooling2D((2, 2)),

                        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.MaxPooling2D((2, 2)),

                        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.MaxPooling2D((2, 2)),

                        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.MaxPooling2D((2, 2)),

                        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
                        tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),

                        tf.keras.layers.Flatten(),  # inputshape = 2*2*512
                        tf.keras.layers.Dense(4096, activation='relu'),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense(4096, activation='relu'),
                        tf.keras.layers.Dropout(0.5),
                        tf.keras.layers.Dense(10, activation='softmax')]
        vgg16_model = tf.keras.Sequential(vgg16_layers)
        vgg16_model.summary()

    # VGG原文中采用带动量的SGD作为优化器，初始学习率为0.01，每次下降为原来的十分之一
    sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
    # 使用可变学习率函数，让调度器去调整
    change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

    if vgg16_reloadstate == False:
        vgg16_model.compile(loss='sparse_categorical_crossentropy',
                            optimizer=sgd, metrics=['accuracy'])
        vgg16_model.fit(train_images, train_labels,
                        epochs=epoch_num,
                        callbacks=[change_lr, cp_callback, csvlog, tensorboard],
                        validation_data=(test_images, test_labels))
        large_loss, large_acc = vgg16_model.evaluate(x=test_images, y=test_labels, verbose=0)
    else:
        vgg16_reload.fit(train_images, train_labels,
                         epochs=epoch_num - ckpt_num,
                         callbacks=[change_lr, cp_callback, csvlog, tensorboard],
                         validation_data=(test_images, test_labels))
        large_loss, large_acc = vgg16_reload.evaluate(x=test_images, y=test_labels, verbose=0)


    print('\nCIFAR10 VGG16 val_loss/accurary:', large_loss, large_acc)
    # print('\nCIFAR100 VGG16 val_loss/accurary:', large_loss, large_acc)

    # 保存模型
    if vgg16_reloadstate == False:
        vgg16_model.save('cifar10_vgg16.h5')
        # vgg16_model.save('cifar100_vgg16.h5')
    else:
        vgg16_reload.save('cifar10_vgg16.h5')
        # vgg16_reload.save('cifar100_vgg16.h5')