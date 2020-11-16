#coding=utf-8
import numpy as np
import operator
import os
import copy
from matplotlib.font_manager import FontProperties
from scipy.interpolate import lagrange
import random
import matplotlib.pyplot as plt
import math

np.set_printoptions(suppress=True)
# 把opt文件内的逗号变为空格
# 数据在我的百度云数据库txt文件,及opt文件
np.set_printoptions(threshold=np.inf)  # 输出全部矩阵不带省略号
random.seed(10)
##########################################
data = np.loadtxt('final37.txt')
# data = data[0:10000]#抽取一部分
x1 = data[:, 5]  # x起点坐标
x2 = data[:, 9]  # x终点坐标
y1 = data[:, 6]  # y起
y2 = data[:, 10]  # y起
z1 = data[:, 4]  # IDpart
z2 = data[:, 8]  # IDpart
diam = data[:, 12]
s1 = [a1 for a1 in range(1, len(x1) - 1) if z1[a1] == z2[a1 - 1] != -1 or z1[a1] != z2[a1 - 1]]  # id相同不等于0，或id不同
# print(s1)
lx = []  # x1,x2相同的部分组成的列表
lxqi = []
lxzg = []
for i1 in range(len(s1) - 1):
    b1 = x1[s1[i1]:s1[i1 + 1]]
    b1 = b1.tolist()
    b2 = x2[s1[i1 + 1] - 1]  # s1[i1]相当于a1
    #     b1 = b1 + [b2]#把与x2最后相连的一个数和x1拼接起来
    b5 = z1[s1[i1]]  # x,y起点id
    b1qi_id = [b5] + b1 + [b2]
    b6 = z2[s1[i1 + 1] - 1]  # x,y终点id
    b1zg_id = [b6] + b1 + [b2]
    lx.append(b1)
    lxqi.append(b1qi_id)
    lxzg.append(b1zg_id)
###################################################
ly = []  # y坐标以及管径大小
for i3 in range(len(s1) - 1):
    b3 = y1[s1[i3]:s1[i3 + 1]]
    b3 = b3.tolist()
    b4 = y2[s1[i3 + 1] - 1]  # y最后一个不相等的数
    b3 = b3 + [b4]
    dm = diam[s1[i3 + 1] - 1]
    b3 = b3 + [dm]  # 加上管径
    ly.append(b3)
#####################################################
# 带有起点id的x坐标与y坐标合并
for q1 in range(len(lxqi)):
    for q2 in range(len(ly[q1])):
        lxqi[q1].append(ly[q1][q2])
# 带有终点id的x坐标与y坐标合并
for p1 in range(len(lxzg)):
    for p2 in range(len(ly[p1])):
        lxzg[p1].append(ly[p1][p2])
lxqi.sort(key=operator.itemgetter(0))  # 排序，只按照第一个索引大小排序
tou = lxqi
lxzg.sort(key=operator.itemgetter(0))
wei = lxzg
# #########################################
toudeng = []
weideng = []
for dwei in wei:
    for i in range(len(tou) - 1):
        if dwei[0] == tou[i][0] and dwei[0] == tou[i + 1][0]:
            toud = [dwei, tou[i], tou[i + 1]]
            toudeng.append(toud)
for dtou in tou:
    for i in range(len(wei) - 1):
        if dtou[0] == wei[i][0] and dtou[0] == wei[i + 1][0]:
            weid = [wei[i], wei[i + 1], dtou]
            weideng.append(weid)
# ###################################################
datatoudeng = []
dataweideng = []
# 去掉起点id
for i in range(len(toudeng)):
    a = toudeng[i][0][1::]
    b = toudeng[i][1][1::]
    c = toudeng[i][2][1::]
    d = [a] + [b] + [c]
    datatoudeng.append(d)
for i in range(len(weideng)):
    a1 = weideng[i][0][1::]
    b1 = weideng[i][1][1::]
    c1 = weideng[i][2][1::]
    d1 = [a1] + [b1] + [c1]
    dataweideng.append(d1)
# print(dataweideng)
####################################################################
# 判断管径信息是否加进列表，若未加进则只为x,y坐标，为偶数
for i in range(len(dataweideng)):
    a = dataweideng[i]
    assert len(a[0]) % 2 == 1
    assert len(a[1]) % 2 == 1
    assert len(a[2]) % 2 == 1
for i in range(len(datatoudeng)):
    a = datatoudeng[i]
    assert len(a[0]) % 2 == 1
    assert len(a[1]) % 2 == 1
    assert len(a[2]) % 2 == 1
finaldata = datatoudeng + dataweideng  # 未插值
final = datatoudeng  # 所有分叉，头等分叉，尾等分叉
final = np.array(final)
#############################################################
# 主分支最后两个数相等在这里把它们删除
for i in range(len(final)):
    # 处理主分支
    len_zhu = len(final[i][0])
    len_zhu_x = len_zhu // 2
    del final[i][0][len_zhu_x - 1]
    len_zhu = len(final[i][0])
    del final[i][0][len_zhu - 2]
final = final.tolist()


##############################################################
# 计算两点之间距离
def get_len(x1, x2, y1, y2):
    diff_x = (x1 - x2) ** 2
    diff_y = (y1 - y2) ** 2
    length = np.sqrt(diff_x + diff_y)
    return length


# 余弦定理计算角度公式
def cal_angle(a, b, c):
    cos_angle = (a ** 2 + b ** 2 - c ** 2) / (2 * a * b)
    angle = np.arccos(cos_angle)
    angle = angle * 180 / np.pi
    return angle


###############################################################################
def Manage_gen(gen_imgs):
    # gen_imgs一个维度为(-1,3,60)的数组，头部分支的尾部，与左右分支的头部分开了
    # 目的把头的尾部，加入左右分支头部，并保证，维度不变
    gen_imgs = gen_imgs[:, :, 0:61]
    finaldata = gen_imgs.tolist()
    final = []
    for i in range(len(finaldata)):
        zhu = finaldata[i][0]
        zuo = finaldata[i][1]
        you = finaldata[i][2]
        # 单独分开x,y,列表
        zhu_x = zhu[0:30]
        zhu_y = zhu[30:60]
        zhu_diam = [zhu[-1]]
        zuo_x = zuo[0:30]
        zuo_y = zuo[30:60]
        zuo_diam = [zuo[-1]]
        you_x = you[0:30]
        you_y = you[30:60]
        you_diam = [you[-1]]
        ############################################
        # 真实数据主分支最后两个基本相等，所以生成数据也是，这样计算角度时，就应该计算最后一个和倒数第三个点
        # 为了让主分支最后一个加在左右分支的头部，此处先去掉左右分支的最后一个点，因为端点插入的值都是相等的，所以去掉影响不大
        # 然后，再将主分支的尾部，加入左右分支头部，这样，就保证了维度不变
        # 去除左右分支尾部一个数
        del zuo_x[-1]
        del zuo_y[-1]
        del you_x[-1]
        del you_y[-1]
        # 在左右分支的头部插入主分支的尾部
        zuo_x.insert(0, zhu_x[-1])
        zuo_y.insert(0, zhu_y[-1])
        you_x.insert(0, zhu_x[-1])
        you_y.insert(0, zhu_y[-1])
        zhu_x.extend(zhu_y)
        zuo_x.extend(zuo_y)
        you_x.extend(you_y)
        zhu_x.extend(zhu_diam)
        zuo_x.extend(zuo_diam)
        you_x.extend(you_diam)
        fencha = [zhu_x] + [zuo_x] + [you_x]
        final.append(fencha)
    final = np.array(final)  # 一个维度为(-1,3,61)的数组
    return final


# 计算端点长度
def get_duandian_len(zhu_x, zhu_y, zuo_x, zuo_y, you_x, you_y):
    # 主分支头尾坐标
    zhu_x_tou = zhu_x[0]
    zhu_y_tou = zhu_y[0]
    zhu_x_wei = zhu_x[-1]
    zhu_y_wei = zhu_y[-1]
    # 左分支头尾坐标
    zuo_x_tou = zuo_x[0]
    zuo_y_tou = zuo_y[0]
    zuo_x_wei = zuo_x[-1]
    zuo_y_wei = zuo_y[-1]
    # 右分支头尾坐标
    you_x_tou = you_x[0]
    you_y_tou = you_y[0]
    you_x_wei = you_x[-1]
    you_y_wei = you_y[-1]
    # 主分支端点长
    zhu_duan_len = get_len(zhu_x_tou, zhu_x_wei, zhu_y_tou, zhu_y_wei)
    # 左分支端点长
    zuo_duan_len = get_len(zuo_x_tou, zuo_x_wei, zuo_y_tou, zuo_y_wei)
    # 右分支端点长
    you_duan_len = get_len(you_x_tou, you_x_wei, you_y_tou, you_y_wei)
    return zhu_duan_len, zuo_duan_len, you_duan_len


# 计算分支总长度
def get_total_len(zhu_x, zhu_y, zuo_x, zuo_y, you_x, you_y):
    zhu_lin_list = []
    zuo_lin_list = []
    you_lin_list = []
    for i in range(1, len(zhu_x)):
        # 相邻点大小相差1e-5可以认为他们是插入点，距离为0
        threshold = 0
        zhu_lin_len = get_len(zhu_x[i - 1], zhu_x[i], zhu_y[i - 1], zhu_y[i])
        if zhu_lin_len < threshold:
            zhu_lin_len = 0
        zuo_lin_len = get_len(zuo_x[i - 1], zuo_x[i], zuo_y[i - 1], zuo_y[i])
        if zuo_lin_len < threshold:
            zuo_lin_len = 0
        you_lin_len = get_len(you_x[i - 1], you_x[i], you_y[i - 1], you_y[i])
        if you_lin_len < threshold:
            you_lin_len = 0
        zhu_lin_list.append(zhu_lin_len)
        zuo_lin_list.append(zuo_lin_len)
        you_lin_list.append(you_lin_len)
    zhu_total_len = 0
    for file in zhu_lin_list:
        zhu_total_len += file
    zuo_total_len = 0
    for file in zuo_lin_list:
        zuo_total_len += file
    you_total_len = 0
    for file in you_lin_list:
        you_total_len += file
    return zhu_total_len, zuo_total_len, you_total_len


# 计算角度
def get_angle(zhu_x, zhu_y, zuo_x, zuo_y, you_x, you_y):
    # 主分支上两个尾部点
    zhu_x_a = zhu_x[-2]
    zhu_y_a = zhu_y[-2]
    zhu_x_b = zhu_x[-1]
    zhu_y_b = zhu_y[-1]
    # 左分支上两个头部点
    zuo_x_a = zuo_x[0]
    zuo_y_a = zuo_y[0]
    zuo_x_b = zuo_x[1]
    zuo_y_b = zuo_y[1]
    # 右分支上两个头部点
    you_x_a = you_x[0]
    you_y_a = you_y[0]
    you_x_b = you_x[1]
    you_y_b = you_y[1]
    # zhu_x_b,zuo_x_a,you_x_a应该相等
    # 每个端点两点长度
    zhu_ab_len = get_len(zhu_x_a, zhu_x_b, zhu_y_a, zhu_y_b)
    zuo_ab_len = get_len(zuo_x_a, zuo_x_b, zuo_y_a, zuo_y_b)
    you_ab_len = get_len(you_x_a, you_x_b, you_y_a, you_y_b)
    zhu_zuo_len = get_len(zhu_x_a, zuo_x_b, zhu_y_a, zuo_y_b)
    zhu_you_len = get_len(zhu_x_a, you_x_b, zhu_y_a, you_y_b)
    zuo_you_len = get_len(zuo_x_b, you_x_b, zuo_y_b, you_y_b)
    zhu_zuo_angle = cal_angle(zhu_ab_len, zuo_ab_len, zhu_zuo_len)
    zhu_you_angle = cal_angle(zhu_ab_len, you_ab_len, zhu_you_len)
    zuo_you_angle = cal_angle(zuo_ab_len, you_ab_len, zuo_you_len)
    zong_angle = zhu_zuo_angle + zhu_you_angle + zuo_you_angle
    return zhu_zuo_angle, zhu_you_angle, zuo_you_angle, zong_angle


# 计算卷曲度
def get_juanqu(zhu_duan_len, zuo_duan_len, you_duan_len, zhu_total_len, zuo_total_len, you_total_len):
    zhu_juanqu = zhu_total_len / zhu_duan_len
    zuo_juanqu = zuo_total_len / zuo_duan_len
    you_juanqu = you_total_len / you_duan_len
    return zhu_juanqu, zuo_juanqu, you_juanqu


######################################################################
# 插值方法三：在相邻两点坐标距离最大的地方插值
finaldata = []
for i in range(len(final)):
    zhu = final[i][0]
    zuo = final[i][1]
    you = final[i][2]
    zhu_diam = [zhu[-1]]
    zuo_diam = [zuo[-1]]
    you_diam = [you[-1]]
    zhu_x = zhu[0:len(zhu) // 2]
    zuo_x = zuo[0:len(zuo) // 2]
    you_x = you[0:len(you) // 2]
    zhu_y = zhu[len(zhu) // 2:(len(zhu) - 1)]
    zuo_y = zuo[len(zuo) // 2:(len(zuo) - 1)]
    you_y = you[len(you) // 2:(len(you) - 1)]
    #     plt.plot(zhu_x,zhu_y,color='red')
    #     plt.plot(zuo_x,zuo_y,color='blue')
    #     plt.plot(you_x,you_y,color='green')
    while len(zhu_x) < 30:
        zhu_lin_list = []
        for j in range(1, len(zhu_x)):
            zhu_lin_len = get_len(zhu_x[j - 1], zhu_x[j], zhu_y[j - 1], zhu_y[j])
            zhu_lin_list.append(zhu_lin_len)
        zhu_max_index = zhu_lin_list.index(max(zhu_lin_list))  # j-1
        # 不处理的话会出现nan
        if abs(zhu_x[zhu_max_index] - zhu_x[zhu_max_index + 1]) == 0:
            zhu_x[zhu_max_index + 1] = zhu_x[zhu_max_index + 1] + 1
        zhu_insert_x = np.linspace(zhu_x[zhu_max_index], zhu_x[zhu_max_index + 1], 3)
        # 插入的点
        zhu_insert_x = zhu_insert_x[1]
        f_zhu = lagrange([zhu_x[zhu_max_index], zhu_x[zhu_max_index + 1]],
                         [zhu_y[zhu_max_index], zhu_y[zhu_max_index + 1]])
        zhu_insert_y = f_zhu(zhu_insert_x)
        zhu_x.insert(zhu_max_index + 1, zhu_insert_x)
        zhu_y.insert(zhu_max_index + 1, zhu_insert_y)
    while len(zuo_x) < 31:
        zuo_lin_list = []
        for j in range(1, len(zuo_x)):
            zuo_lin_len = get_len(zuo_x[j - 1], zuo_x[j], zuo_y[j - 1], zuo_y[j])
            zuo_lin_list.append(zuo_lin_len)
        zuo_max_index = zuo_lin_list.index(max(zuo_lin_list))  # 对应j-1
        if abs(zuo_x[zuo_max_index] - zuo_x[zuo_max_index + 1]) == 0:
            zuo_x[zuo_max_index + 1] = zuo_x[zuo_max_index + 1] + 1
        zuo_insert_x = np.linspace(zuo_x[zuo_max_index], zuo_x[zuo_max_index + 1], 3)
        #         #插入的点
        zuo_insert_x = zuo_insert_x[1]
        f_zuo = lagrange([zuo_x[zuo_max_index], zuo_x[zuo_max_index + 1]],
                         [zuo_y[zuo_max_index], zuo_y[zuo_max_index + 1]])
        zuo_insert_y = f_zuo(zuo_insert_x)
        zuo_x.insert(zuo_max_index + 1, zuo_insert_x)
        zuo_y.insert(zuo_max_index + 1, zuo_insert_y)
    while len(you_x) < 31:
        you_lin_list = []
        for j in range(1, len(you_x)):
            # 计算相邻两坐标的距离
            you_lin_len = get_len(you_x[j - 1], you_x[j], you_y[j - 1], you_y[j])
            # 添加进列表中
            you_lin_list.append(you_lin_len)
        # 计算距离最大的值对应的索引，对应x坐标的j-1和j之间的距离，最大
        you_max_index = you_lin_list.index(max(you_lin_list))  # 对应j-1
        # 然后，在两个最大点之间，平均插入一个数，作为插入x点
        if abs(you_x[you_max_index] - you_x[you_max_index + 1]) == 0:
            you_x[you_max_index + 1] = you_x[you_max_index + 1] + 1
        you_insert_x = np.linspace(you_x[you_max_index], you_x[you_max_index + 1], 3)
        # 插入的点
        you_insert_x = you_insert_x[1]
        # 拉格朗日计算直线方程
        f_you = lagrange([you_x[you_max_index], you_x[you_max_index + 1]],
                         [you_y[you_max_index], you_y[you_max_index + 1]])
        # 插入的y点
        you_insert_y = f_you(you_insert_x)
        # 将求得的x,y点插入对应位置
        you_x.insert(you_max_index + 1, you_insert_x)
        you_y.insert(you_max_index + 1, you_insert_y)
    ################################################
    # 可视化插值点
    #     plt.scatter(zhu_x,zhu_y,marker='*',color='red')
    #     plt.scatter(zuo_x,zuo_y,marker='*',color='blue')
    #     plt.scatter(you_x,you_y,marker='*',color='green')
    #     plt.show()
    #####################################################################
    # 主分支最后两个相等的前面已经删除，现在删除左右分支第一个
    zuo_x = zuo_x[1::]
    you_x = you_x[1::]
    zuo_y = zuo_y[1::]
    you_y = you_y[1::]
    # 这里将x和y列表再接起来
    zhu_xy = zhu_x + zhu_y
    zuo_xy = zuo_x + zuo_y
    you_xy = you_x + you_y
    # 这里再将坐标点与管径接起来
    zhu = zhu_xy + zhu_diam
    zuo = zuo_xy + zuo_diam
    you = you_xy + you_diam
    fencha = [zhu] + [zuo] + [you]
    finaldata.append(fencha)
final = np.array(finaldata)  # 数据维度为(-1,3,61)


# data = final.reshape(-1,3,61)
# for i in range(len(data)):
#         plt.scatter(data[i][0][0:30],data[i][0][30:60],marker='*',color='red')
#         plt.scatter(data[i][1][0:30],data[i][1][30:60],marker='*',color='blue')
#         plt.scatter(data[i][2][0:30],data[i][2][30:60],marker='*',color='green')
#
#         plt.plot(data[i][0][0:30],data[i][0][30:60],color='red')
#         plt.plot(data[i][1][0:30],data[i][1][30:60],color='blue')
#         plt.plot(data[i][2][0:30],data[i][2][30:60],color='green')
#         plt.show()
##################################################################
def rotate(angle, valuex, valuey):
    rotatex = math.cos(angle) * valuex - math.sin(angle) * valuey
    rotatey = math.cos(angle) * valuey + math.sin(angle) * valuex
    return rotatex, rotatey


# 测试一张图片
# final = final[0:1,:,:]
final1 = []
for i in range(len(final)):
    x = final[i, :, 0:30]
    y = final[i, :, 30:60]
    diam = final[i, :, -1]
    diam = diam.reshape(3, 1)
    rotatedata = []
    for j in range(0, 360, 2):  # 每隔3，30度旋转一次
        x1, y1 = rotate(j, x, y)
        x1_mean = np.mean(x1)
        y1_mean = np.mean(y1)
        cha_mean = x1_mean - y1_mean
        if (cha_mean > 0):
            y1 += abs(cha_mean)
        else:
            x1 += abs(cha_mean)
        x1_min = np.min(x1)
        y1_min = np.min(y1)
        x1 = x1 - x1_min
        y1 = y1 - y1_min
        rotate_final = np.concatenate((x1, y1, diam), axis=1)
        rotatedata.append(rotate_final)
    final1.append(rotatedata)
finaldata = []
for file in final1:
    for data in file:
        finaldata.append(data)
final = np.array(finaldata)
#####################################################
finalxy = final[:, :, 0:60]
finaldiam = final[:, :, -1]
finaldiam = finaldiam.reshape(-1, 3, 1)
finalxy_min = np.min(finalxy)
finalxy_max = np.max(finalxy)
final1 = []
final2 = []
final3 = []
final4 = []
for i in range(len(final)):
    final_max = np.max(final[i])
    if final_max <= (finalxy_max / 4):
        final1.append(final[i])
    elif (finalxy_max / 4) < final_max <= (finalxy_max * 2 / 4):
        final2.append(final[i])
    elif (finalxy_max * 2 / 4) < final_max <= (finalxy_max * 3 / 4):
        final3.append(final[i])
    else:
        final4.append(final[i])
final1 = np.array(final1)  # 最小
final2 = np.array(final2)
final3 = np.array(final3)
final4 = np.array(final4)  # 最大分叉
########################################################################
final = final1
finalxy = final[:, :, 0:60]
finaldiam = final[:, :, -1]
finaldiam = finaldiam.reshape(-1, 3, 1)
finalxy_min = np.min(finalxy)
finalxy_max = np.max(finalxy)
normxy = (finalxy - finalxy_min) / (finalxy_max - finalxy_min)
final = np.concatenate((normxy, finaldiam), 2)
finaldata = []
for i in range(len(final)):
    final_x = final[i][:, 0:30]
    final_y = final[i][:, 30:60]
    final_diam = final[i][:, -1]
    final_diam = final_diam.reshape(3, 1)
    translation_x = (1. - np.max(final_x)) / 2
    translation_y = (1. - np.max(final_y)) / 2
    final_x = final_x + translation_x
    final_y = final_y + translation_y
    finalxyd = np.concatenate((final_x, final_y, final_diam), 1)
    finaldata.append(finalxyd)
finaldata = np.array(finaldata)
np.random.shuffle(finaldata)
print(finaldata.shape)
############################################################################
# #将左右分的头部与主分支尾部相连(-1,3,61)
# finaldata = Manage_gen(finaldata)
# #保存单张图片到文件夹
# data_images=finaldata
# for i in range(len(finaldata)):
#     plt.figure(figsize=(128,128),dpi=1)
#     plt.plot(data_images[i][0][0:30],data_images[i][0][30:60],color='blue',linewidth=np.log(data_images[i][0][-1])*88)
#     plt.plot(data_images[i][1][0:30],data_images[i][1][30:60],color='red',linewidth=np.log(data_images[i][1][-1])*88)
#     plt.plot(data_images[i][2][0:30],data_images[i][2][30:60],color='green',linewidth=np.log(data_images[i][2][-1])*88)
#     plt.axis('off')
#     plt.savefig('C:\\Users\\Administrator\\Desktop\\重新整理血管网络\\未旋转原始图\\original%d.jpg' %(i),dpi=1)
#     plt.close()
##############################################################################
# 可视化图像
# data = finaldata.reshape(-1,3,61)
# for i in range(len(data)):
#         plt.scatter(data[i][0][0:30],data[i][0][30:60],marker='*',color='red')
#         plt.scatter(data[i][1][0:30],data[i][1][30:60],marker='*',color='blue')
#         plt.scatter(data[i][2][0:30],data[i][2][30:60],marker='*',color='green')
#         plt.plot(data[i][0][0:30],data[i][0][30:60],color='red')
#         plt.plot(data[i][1][0:30],data[i][1][30:60],color='blue')
#         plt.plot(data[i][2][0:30],data[i][2][30:60],color='green')
#         plt.xlim(0.,1.)
#         plt.ylim(0.,1.)
#         plt.xticks(np.arange(0,1,0.1))
#         plt.yticks(np.arange(0,1,0.1))
# #         plt.savefig('C:\\Users\\Administrator\\Desktop\\重新整理血管网络\\test\\original%d.jpg' %(i))
# #         plt.close()
#         plt.show()
##############################################################################
# 测试各项指标
# finaldata = final
# finaldata = Manage_gen(finaldata)
# def calculate(data):
#     data = data.reshape(-1,3,61)
#     data = data[:,:,0:61]
#     for i in range(0,len(data)):
#         zhu_x = data[i][0][0:30]
#         zhu_y = data[i][0][30:60]
#         zuo_x = data[i][1][0:30]
#         zuo_y = data[i][1][30:60]
#         you_x = data[i][2][0:30]
#         you_y = data[i][2][30:60]
#         #计算每个分叉分支端点长度
#         zhu_duan_len,zuo_duan_len,you_duan_len = get_duandian_len(zhu_x,zhu_y,zuo_x,zuo_y,you_x,you_y)
#         #计算每个分叉分支总长度
#         zhu_total_len,zuo_total_len,you_total_len = get_total_len(zhu_x,zhu_y,zuo_x,zuo_y,you_x,you_y)
#         #计算每个分叉分支角度
#         zhu_zuo_angle,zhu_you_angle,zuo_you_angle,zong_angle= get_angle(zhu_x,zhu_y,zuo_x,zuo_y,you_x,you_y)
#         #计算卷曲度
#         zhu_juanqu,zuo_juanqu,you_juanqu = get_juanqu(zhu_duan_len,zuo_duan_len,you_duan_len,zhu_total_len,zuo_total_len,you_total_len)
#         print("主长度",zhu_duan_len,"左长度",zuo_duan_len,"右长度",you_duan_len)
#         print("主卷曲度",zhu_juanqu,"左卷曲度",zuo_juanqu,"右卷曲度",you_juanqu)
#         print("主角度",zhu_zuo_angle,"左角度",zhu_you_angle,"右角度",zuo_you_angle,"总角度",zong_angle)
#         print("*******************")
# calculate(finaldata)
######################################################
# 一种普通的可视化方法，此时画出来的图端点都连在了原点位置
# finaldata = finaldata.tolist()
# for i in range(len(finaldata)):
#     plt.plot(finaldata[i][0][0:30],finaldata[i][0][30:60],color='red',linewidth=np.log(finaldata[i][0][-1]))
#     plt.plot([finaldata[i][0][29]]+finaldata[i][1][0:30],[finaldata[i][0][59]]+finaldata[i][1][30:60],color='blue',linewidth=np.log(finaldata[i][1][-1]))
#     plt.plot([finaldata[i][0][29]]+finaldata[i][2][0:30],[finaldata[i][0][59]]+finaldata[i][2][30:60],color='green',linewidth=np.log(finaldata[i][2][-1]))
#     plt.xticks(np.arange(0,1,0.1))
#     plt.yticks(np.arange(0,1,0.1))
#     plt.show()
############################################################
# final1 = finaldata[0:500,:,:]
# final2 = finaldata[1500:2000,:,:]
# np.save('C:\\Users\\Administrator\\Desktop\\重新整理血管网络\\原始吖.npy',final1)
np.save('相对大小分叉.npy', finaldata)
###################################################################
# 每100张图片显示在一张图中
# rows,cols = 10, 10
# fig,axs = plt.subplots(rows,cols)
# cnt = 0
# for i in range(rows):
#     for j in range(cols):
#         xy = finaldata[cnt]#第n个分叉图，有三个分支，每个分支21个数
#         for k in range(len(xy)):
#             x = xy[k][0:30]
#             y = xy[k][30:60]
#             axs[i,j].plot(x,y,linewidth=2)
#             axs[i,j].axis('off')
#         cnt +=1
# plt.show()
######################################################################
