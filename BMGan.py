import sys

import matplotlib
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from os import system

tf.set_random_seed(1)
np.random.seed(1)
# 设置超参数
batch_size = 64  # 批量大小
lr_g = 0.00001  # 生成器的学习率
lr_d = 0.00001  # 判别器的学习率
n_ideas = 17  # （生成器）
art_components = 15  # 15  # 在画上画多少个点

# 列表解析式代替了for循环，paint_points.shape=(64,15),
# np.vstack()默认逐行叠加（axis=0）
paint_points = np.vstack([np.linspace(-1, 1, art_components) for _ in range(batch_size)])


def artist_works():
    # a为64个1到2均匀分布抽取的值，shape=(64,1)
    a = np.random.uniform(1, 2, size=batch_size)[:, np.newaxis]
    # a = np.array([1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    # a = a[:, np.newaxis]
    paintings = a * np.power(paint_points, 2) + (a - 1)
    return paintings


# tf.variable_scope('Generator')
# G_in = tf.placeholder(tf.float32, [None, n_ideas])  # 随机的ideals（来源于正态分布）
# G_l1 = tf.layers.dense(G_in, 256, tf.nn.relu)
# G_l2 = tf.layers.dense(G_l1, 128, tf.nn.relu)
# G_out = tf.layers.dense(G_l2, art_components)  # 得到15个点
# tf.variable_scope('Discriminator')
# real_art = tf.placeholder(tf.float32, [None, art_components], name='real_in')  # 接受专家的画
# D_l0 = tf.layers.dense(real_art, 256, tf.nn.relu, name='l')
# D_l1 = tf.layers.dense(D_l0, 128, tf.nn.relu, name='l1')
# prob_artist0 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out')  # 代入专家的画，判别器判断这副画来自于专家的概率
#
# # 再次利用生成器生成的15个点
# D_l2 = tf.layers.dense(G_out, 256, tf.nn.relu, name='l', reuse=True)
# D_l3 = tf.layers.dense(D_l2, 128, tf.nn.relu, name='l1', reuse=True)  # 接受业余画家的画
# prob_artist1 = tf.layers.dense(D_l3, 1, tf.nn.sigmoid, name='out', reuse=True)  # 代入生成的画，判别器判断这副画来自于专家的概率

# x = tf.placeholder(tf.float32, shape=(None, 2))
# w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
# w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))
# a1 = tf.Variable(100)
# a2 = tf.Variable(200)
# a3 = tf.add(a1, a2)
# a = tf.matmul(x, w1)
# y = tf.matmul(a, w2)
# with tf.Session() as sess:
#     init_op = tf.global_variables_initializer()
#     sess.run(init_op)
#     print(sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]]}))
#     print(sess.run(w1))
#     print(sess.run(w2))
#     print(sess.run(a3))
#     b1=sess.run([a3, a2, a2 + a1], {a1: 3000})
#     print(b1)

with tf.variable_scope('Generator'):
    G_in = tf.placeholder(tf.float32, [None, n_ideas])  # 随机的ideals（来源于正态分布）
    G_l1 = tf.layers.dense(G_in, 256, tf.nn.relu)
    G_l2 = tf.layers.dense(G_l1, 128, tf.nn.relu)
    G_out = tf.layers.dense(G_l2, art_components)  # 得到15个点
# print(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

with tf.variable_scope('Discriminator'):
    real_art = tf.placeholder(tf.float32, [None, art_components], name='real_in')  # 接受实际点
    D_l0 = tf.layers.dense(real_art, 256, tf.nn.relu, name='l')
    D_l1 = tf.layers.dense(D_l0, 128, tf.nn.relu, name='l1')
    prob_artist0 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out')  # 代入实际点，判别器判断实际点的概率

    # 再次利用生成器生成的15个点
    D_l2 = tf.layers.dense(G_out, 256, tf.nn.relu, name='l', reuse=True)
    D_l3 = tf.layers.dense(D_l2, 128, tf.nn.relu, name='l1', reuse=True)  # 接受生成点
    prob_artist1 = tf.layers.dense(D_l3, 1, tf.nn.sigmoid, name='out', reuse=True)  # 代入生成的点，判别器判断概率

G_loss = tf.reduce_mean(tf.log(1 - prob_artist1))

D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1 - prob_artist1))  #

train_G = tf.train.AdamOptimizer(lr_g).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

train_D = tf.train.AdamOptimizer(lr_d).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()  # 连续画图
plt.rcParams['font.family'] = 'STSong'
plt.rcParams['font.size'] = 15

for step in range(10000):
    artist_paintings = artist_works()  # 是实际点
    G_ideas = np.random.randn(batch_size, n_ideas)
    G_paintings, pa0, pa1, Dl = sess.run(fetches=[G_out, prob_artist0, prob_artist1, D_loss, train_D, train_G],
                                         feed_dict={G_in: G_ideas, real_art: artist_paintings})[:4]  # 训练和获取结果
    # print('pa0:', pa0)
    # print('pa1:', pa1)
    # print('Dl:', Dl)
    # print(sess.run([-tf.reduce_mean(tf.log(pa0) + tf.log(1 - pa1))]))
    if step % 50 == 0:  # 每50步训练画一次图
        plt.cla()
        plt.plot(paint_points[0], G_paintings[0], c='#4AD631', lw=3, label='预测值', )
        plt.plot(paint_points[0], artist_paintings[0], c='#FAD600', lw=3, label='真实值', )
        plt.plot(paint_points[0], 2 * np.power(paint_points[0],2) + 1, c='#74BCFF', lw=3, label='上限')
        plt.plot(paint_points[0], 1 * np.power(paint_points[0],2) + 0, c='#FF9359', lw=3, label='下限')

        # print('dsfdsfsdfsdfdsfds\n', pa0.mean())
        plt.text(-.5, 2.3, '判别器精度=%.2f (0.5 真实值判别概率)' % pa0.mean(), fontdict={'size': 15})
        plt.text(-.5, 2, '判别器精度=%.2f (0.5 生成器模拟值判别概率)' % pa1.mean(), fontdict={'size': 15})
        # plt.text(-.5, 1.7, 'D score= %.2f (-1.38 for G to converge)' % -Dl, fontdict={'size': 15})
        plt.ylim((0, 3))
        # plt.xlim((-2, 2))
        plt.legend(loc='upper right', fontsize=12)
        plt.draw()
        plt.pause(0.01)
plt.ioff()
plt.show()
