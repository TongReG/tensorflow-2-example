# 基础1 基于tensorflow 1.x API，计算线型回归
import os
import time
import tensorflow.compat.v1 as tf
import numpy as np


global timecnt
randomnum = 2000


def start_train(sess,
                x_vals, y_vals,
                train_step, loss_function):
    flag = 1
    nums = len(x_vals)
    print("Data Length = " + str(nums))
    for i in range(nums):  # 注意循环值是0到nums-1，不包括nums
        # 随机从样本中取值
        rand_index = np.random.choice(nums)
        rand_x = [x_vals[rand_index]]
        rand_y = [y_vals[rand_index]]
        # 损失函数引用的placeholder(直接或间接用的都算), x_data使用样本rand_x， y_target用样本rand_y
        if flag == 1:
            timecnt = time.time()
            flag = 0
        sess.run(train_step, feed_dict={x_data: rand_x, y_target: rand_y})
        # 每50循环打印1次情况
        if i % 50 == 0:
            timef = time.time()
            flag = 1
            print('step: ' + str(i) + ' A = ' + str(sess.run(A)))
            print(
                'loss: ' + str(sess.run(loss_function, feed_dict={x_data: rand_x, y_target: rand_y})))
            print('time: ' + str(timef - timecnt))


if __name__ == "__main__":
    # 在代码中关闭eager运算：
    tf.disable_eager_execution()

    # 样本，输入list(Normal Destribution)。设置x的均值为1, 均方误差为0.02, 数据量为randomnum个
    x_vals = np.random.normal(1, 0.02, randomnum)
    # 设置样本输出y为randomnum个值为10的list
    y_vals = np.repeat(10.0, randomnum)

    x_data = tf.placeholder(shape=[1], dtype=tf.float32)
    y_target = tf.placeholder(shape=[1], dtype=tf.float32)

    # 我们定义的模型，是一个线型函数，即 y = k * x， 也就是my_output = A * x_data
    # x_data将用样本x_vals。我们的目标是，算出A的值。
    # 其实已经能猜出，y都是10.0的话，x均值为1, 那么A应该是10。
    A = tf.Variable(tf.random_normal(shape=[1]))
    my_output = tf.multiply(x_data, A)

    # 损失函数，用的是模型算的值，减去实际值的平方。y_target就是上面的y_vals。
    loss = tf.square(my_output - y_target)

    # 根据tf 1.x的概念，获取session
    sess = tf.Session()

    # 我们采用梯度下降算法，设置学习率为0.02, 可以认为是每次迭代修改A，调整0.02。
    # 比如A初始化为20, 发现不好，于是猜测下一个A为20-0.02
    my_opt = tf.train.GradientDescentOptimizer(learning_rate=0.02)
    my_target = my_opt.minimize(loss)  # 设置目标，使得损失函数达到最小值

    # 开始训练
    start_train(sess, x_vals, y_vals, my_target, my_opt)

    time.sleep(2)

    # 这次，我们改用RMSProp算法作为优化器(Optimizer)
    # 修改了AdaGrad的梯度积累为指数加权的移动平均，使得其在非凸设定下效果更好。
    my_optwo = tf.train.RMSPropOptimizer(learning_rate=0.02)
    my_targetwo = my_optwo.minimize(loss)

    # 在使用RMSPropOptimizer构建训练操作时，有一个create_slots的操作需要初始化一些变量
    # 下面的变量初始化操作要放在构建train_step之后
    init = tf.global_variables_initializer()  # 初始化变量
    sess.run(init)

    # 开始训练
    start_train(sess, x_vals, y_vals, my_targetwo, my_optwo)