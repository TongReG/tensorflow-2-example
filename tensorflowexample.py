#基础1 tensorflow 1.x 线型回归example
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # 这一行注释掉就是使用cpu，不注释就是使用gpu
import tensorflow.compat.v1 as tf
import numpy as np

#在代码中关闭eager运算：
tf.disable_eager_execution()

global timecnt 
randomnum = 2000

# 样本，输入列表(Normal Destribution)，均值为1, 均方误差为0.1, 数据量为100个
x_vals = np.random.normal(1, 0.02, randomnum)
# 样本输出列表， 100个值为10.0的列表
y_vals = np.repeat(10.0, randomnum)

x_data = tf.placeholder(shape=[1], dtype=tf.float32)
y_target = tf.placeholder(shape=[1], dtype= tf.float32)

A = tf.Variable(tf.random_normal(shape=[1]))

# 我们定义的模型，是一个线型函数，即 y = w * x， 也就是my_output = A * x_data
# x_data将用样本x_vals。我们的目标是，算出A的值。
# 其实已经能猜出，y都是10.0的话，x均值为1, 那么A应该是10。哈哈
my_output = tf.multiply(x_data, A)

# 损失函数， 用的是模型算的值，减去实际值， 的平方。y_target就是上面的y_vals。
loss = tf.square(my_output - y_target)

sess = tf.Session()
init = tf.global_variables_initializer()#初始化变量
sess.run(init)

# 梯度下降算法， 学习率0.02, 可以认为每次迭代修改A，修改一次0.02。比如A初始化为20, 发现不好，于是猜测下一个A为20-0.02
my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
train_step = my_opt.minimize(loss)#目标，使得损失函数达到最小值

flag = 1
for i in range(randomnum):#0到100,不包括100
    # 随机从样本中取值
    rand_index = np.random.choice(randomnum)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    #损失函数引用的placeholder(直接或间接用的都算), x_data使用样本rand_x， y_target用样本rand_y
    if flag == 1:
        timecnt = time.time()
        flag = 0
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    #打印
    if i % 50 == 0:
        timef = time.time()
        flag = 1
        print('step: ' + str(i) + ' A = ' + str(sess.run(A)))
        print('loss: ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))
        print('time: ' + str(timef - timecnt))

time.sleep(2)

#RMSProp算法 修改了AdaGrad的梯度积累为指数加权的移动平均，使得其在非凸设定下效果更好。
my_optwo = tf.train.RMSPropOptimizer(learning_rate=0.02)
train_step = my_optwo.minimize(loss)

#在使用RMSPropOptimizer这个优化器构建训练op时，有一个create_slots的操作需要初始化一些变量
#下面的变量初始化操作要放在构建train_step之后
init = tf.global_variables_initializer()#初始化变量
sess.run(init)

flag = 1
for i in range(randomnum):#0到100,不包括100
    # 随机从样本中取值
    rand_index = np.random.choice(randomnum)
    rand_x = [x_vals[rand_index]]
    rand_y = [y_vals[rand_index]]
    if flag == 1:
        timecnt = time.time()
        flag = 0
    sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
    #打印
    if i % 50 == 0:
        timef = time.time()
        flag = 1
        print('step: ' + str(i) + ' A = ' + str(sess.run(A)))
        print('loss: ' + str(sess.run(loss, feed_dict={x_data: rand_x, y_target: rand_y})))
        print('time: ' + str(timef - timecnt))