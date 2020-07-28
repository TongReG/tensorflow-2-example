# CIFAR1O + VGG13/16
import os
import time
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用cpu，不注释就是使用gpu
# tf.random.set_seed(2345)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(device=physical_devices[0],
# enable=True)

#预处理
def preprocess(x,y):
    x = tf.cast(x,dtype=tf.float32) / 255.
    y = tf.cast(y,dtype=tf.int32)
    return x,y

batchsize = 64
(train_images, train_labels),(test_images, test_labels) = tf.keras.datasets.cifar10.load_data()
train_label_s = tf.squeeze(train_labels,axis=1)
test_label_s = tf.squeeze(test_labels,axis=1)
# print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
# batch就是将多个元素组合成batch
# shuffle的功能为打乱dataset中的元素，参数buffersize表示打乱时使用的buffer的大小
train_data = tf.data.Dataset.from_tensor_slices((train_images, train_label_s))
train_data = train_data.shuffle(buffer_size=5000).map(preprocess).batch(batchsize)

#tf.keras.utils.to_categorical将数据转换为one hot格式
#train_labels = tf.keras.utils.to_categorical(train_labels, 10)
#test_labels = tf.keras.utils.to_categorical(test_labels, 10)
test_data = tf.data.Dataset.from_tensor_slices((test_images, test_label_s))
test_data = test_data.map(preprocess).batch(batchsize)

#这一部分打印train_data的信息
sample = next(iter(train_data))
print('VGG13 BatchSize =',batchsize)
print('sample:',sample[0].shape,sample[1].shape,
      tf.reduce_min(sample[0]),tf.reduce_max(sample[0]))

vgg13_layers = [tf.keras.layers.Conv2D(64,kernel_size=[3,3],padding="same",activation=tf.nn.relu),
    tf.keras.layers.Conv2D(64, kernel_size=[3, 3], padding="same", activation=tf.nn.relu),
    tf.keras.layers.MaxPool2D(pool_size=[2,2],strides=2,padding="same"),

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
    # x=tf.random.normal([4,32,32,3])
    # out=conv_net(x)
    # print(out.shape)
fc_net = tf.keras.Sequential([tf.keras.layers.Dense(4096,activation=tf.nn.relu),
        tf.keras.layers.Dense(4096,activation=tf.nn.relu),
        tf.keras.layers.Dense(10,activation='softmax'),])
vgg13_net.build(input_shape=[None, 32, 32, 3])
fc_net.build(input_shape=[None,512])
optimizer = tf.keras.optimizers.Adam(lr=1e-4)

#tf.trainable_variables()函数可以也仅可以查看可训练的变量
variables = vgg13_net.trainable_variables + fc_net.trainable_variables
flag = 1
for epoch in range(50):
    for step,(x,y) in enumerate(train_data): #one epoch has 50000 photos steps = 50000/batchsize
        if flag == 1:
            start = time.clock()
            flag = 0
        with tf.GradientTape() as tape:
            out = vgg13_net(x)
            out = tf.reshape(out,[-1,512])
            logits = fc_net(out)
            y_onehot = tf.one_hot(y,depth=10)
            loss = tf.losses.categorical_crossentropy(y_onehot,logits,from_logits=True)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss,variables)   #计算梯度
        optimizer.apply_gradients(zip(grads,variables))  #更新梯度
        if step % 16 == 0:
            elapsed = (time.clock() - start)
            flag = 1
            print('Epoch:',epoch,'Step:',step,'datas:',step * batchsize,'loss:',float(loss))
            print('Time:',elapsed)

    total_num = 0
    total_correct = 0
    for x,y in test_data:
        out = vgg13_net(x)
        out = tf.reshape(out,[-1,512])
        logits = fc_net(out)
        prob = tf.nn.softmax(logits,axis=1)
        pred = tf.argmax(prob,axis=1)
        pred = tf.cast(pred,dtype=tf.int32)
        correct = tf.cast(tf.equal(pred,y),dtype=tf.int32)
        correct = tf.reduce_sum(correct)
        total_num+=x.shape[0]
        total_correct+=int(correct)
    acc = total_correct / total_num
    print('Epoch:',epoch,'acc:',acc)


weight_decay = 5e-4
dropout_rate = 0.5
batch_size = 128
learning_rate = 1e-2
epoch_num = 50
vgg16_layers = [tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3), kernel_regularizer=tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.MaxPooling2D((2, 2)),

    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    tf.keras.layers.Conv2D(512, (3, 3), activation='relu', padding='same', kernel_regularizer = tf.keras.regularizers.l2(weight_decay)),
    
    tf.keras.layers.Flatten(),  # inputshape = 2*2*512
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4096, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')]
vgg16_model = tf.keras.Sequential(vgg16_layers)

#变学习率的设置方式。要用到的是model.fit中的callbacks参数，从参数名可以理解，我们需要写一个回调函数来实现学习率随训练轮数增加而减小。
#VGG原文中采用带动量的SGD，初始学习率为0.01，每次下降为原来的十分之一
#这里我们让网络训练50个epoch，即epoch_num = 50
#其中前20个采用0.01，中间20个采用0.001，最后10个采用0.0001
def scheduler(epoch):
    if epoch < epoch_num * 0.4:
        return learning_rate
    if epoch < epoch_num * 0.8:
        return learning_rate * 0.1
    return learning_rate * 0.01

sgd = tf.keras.optimizers.SGD(lr=learning_rate, momentum=0.9, nesterov=True)
change_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)

vgg16_model.compile(loss='sparse_categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
vgg16_model.fit(train_images, train_labels,
          epochs=epoch_num,
          callbacks=[change_lr],
          validation_data=(test_images, test_labels))

test_loss, test_acc = vgg13_net.evaluate(x=test_images, y=test_labels, verbose=0)
large_loss, large_acc = vgg16_model.evaluate(x=test_images, y=test_labels, verbose=0)
print('\nCIFAR10 VGG13 val_loss/accurary:' , test_loss, test_acc)
print('\nCIFAR10 VGG16 val_loss/accurary:', large_loss, large_acc)
