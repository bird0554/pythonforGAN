# coding=utf-8
import numpy as np
import matplotlib.pyplot as plt


def bootstrap_replicate_1d(data, func):
    bs_sample = np.random.choice(data, len(data))
    return func(bs_sample)

# 产生一次bootstrap sample，并对这个sample进行一次func操作
# ————————————————
# 版权声明：本文为CSDN博主「刀尔東」的原创文章，遵循 CC 4.0 BY-SA 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https://blog.csdn.net/weixin_38760323/article/details/81265843

def draw_bs_reps(data, func, size=1):
    bs_replicates = np.empty(size)
    for i in range(size):
        bs_replicates[i] = bootstrap_replicate_1d(data, func)
    return bs_replicates

# 重复size次bootstrap_replicate_1d
def ecdf(data):
    """Compute ECDF for a one-dimensional array of measurements."""
    # Number of data points: n
    n = len(data)
    # x-data for the ECDF: x
    x = np.sort(data)
    # y-data for the ECDF: y
    y = np.arange(1, n + 1) / float(n)
    print x, y
    return x, y

for i in range(50):  # 做50次实验
    # Generate bootstrap sample: bs_sample
    rainfall = np.array(
        [-22, -3, 100, -23, -56. - 90, 11, 112, 234, 55, 556, 67.4, 455, 89, 33, 43, 332, 44.8, 11, 223, 234, 5, 556,
         67, 95, 89, 3, 33.5, 342, 44], dtype=float)
    # rainfall=np.array(rainfall)
    # rainfall=rainfall.reshape(2,6)
    bs_sample = np.random.choice(rainfall, size=len(rainfall))  # 每次有放回地抽取一个和rainfall一样长的样本bs_sample

    # Compute and plot ECDF from bootstrap sample
    x, y = ecdf(bs_sample)  # 为bs_sample生成一个CDF（累计密度函数），返回x值是降雨量，y值是小于这个x值的概率
    _ = plt.plot(x, y, marker='.', linestyle='none',
                 c='gray', alpha=0.1)  # 将这次循环中的bs_sample(i)画在图上

# Compute and plot ECDF from original data
x, y = ecdf(rainfall)  # 为真实值生成一个CDF
_ = plt.plot(x, y, marker='.')  # 也画在图上

# Make margins and label axes
plt.margins(0.02)
_ = plt.xlabel('yearly rainfall (mm)')
_ = plt.ylabel('ECDF')
plt.show()

# Take 10,000 bootstrap replicates of the mean: bs_replicates
bs_replicates = draw_bs_reps(rainfall, np.mean, size=10000)  # 将rainfall做bootstrap sample，并将结果均值，重复10000次
conf_int = np.percentile(bs_replicates,[2.5,97.5])

bs_replicates1= draw_bs_reps(rainfall, np.mean, size=10000)
conf_int1 = np.percentile(bs_replicates1,[2.5,97.5])
print '##################'
print conf_int

print conf_int1
print '##################'

# Compute and print SEM
sem = np.std(rainfall) / np.sqrt(len(rainfall))  # 求出rainfall真实值的标准差
print(sem)
# Compute and print standard deviation of bootstrap replicates
bs_std = np.std(bs_replicates)  # 输出10000次平均值的标准差
print(bs_std)

# Make a histogram of the results
_ = plt.hist(bs_replicates, bins=50, normed=True)  # 做10000次实验的概率直方图
_ = plt.xlabel('mean annual rainfall (mm)')
_ = plt.ylabel('PDF')

# Show the plot
plt.show()

# Show the plot
# ————————————————
# 版权声明：本文为CSDN博主「刀尔東」的原创文章，遵循
# CC
# 4.0
# BY - SA
# 版权协议，转载请附上原文出处链接及本声明。
# 原文链接：https: // blog.csdn.net / weixin_38760323 / article / details / 81265843
