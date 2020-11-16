# coding=utf-8
from  numpy import *
from sklearn.preprocessing import StandardScaler


def contrib(matr, alpha=0.8):
    # li = [[1, 1], [1, 3], [2, 2], [4, 3], [2, 3]]
    # matrix = mat(li)
    std = StandardScaler()
    matr = std.fit_transform(matr)
    matrix = mat(matr).astype(dtype='float32')
    # print 'matrix:'
    # print matrix
    # 求均值
    # mean_matrix = mean(matrix, axis=0)
    # print(mean_matrix.shape)
    # 减去平均值
    # Dataadjust = matrix - mean_matrix
    # print(Dataadjust.shape)
    # 计算特征值和特征向量
    covMatrix = cov(matrix, rowvar=0)
    print('covMatrix')
    print(covMatrix)
    covMatrix1 = corrcoef(matrix, rowvar=0)
    print('covMatrix1:')
    print(covMatrix1)
    eigValues, eigVectors = linalg.eig(covMatrix)
    eigValues = eigValues.real
    eigVectors = eigVectors.real

    # print 'eigValues:'
    # print eigValues
    # return eigValues.real,eigVectors.real
    # print(eigValues.shape)
    # print(eigVectors.shape)
    # '''计算各主成分方差贡献'''
    print('@@@@@@@@@@@@@@')
    print(eigValues)
    contribute = [eigValues[i] / sum(eigValues) for i in range(len(eigValues))]
    # '''保存特征值排序后与之前对应的位置'''
    sort = argsort(contribute)

    # '''根据传入的累计贡献率阈值alpha提取所需的主成分'''
    pca = []
    token = 0
    i = 1
    while (token <= alpha):
        token = token + contribute[sort[len(contribute) - i]]
        pca.append(sort[len(contribute) - i])
        i += 1

    # '''将得到的各主成分对应的特征值和特征向量保存下来并作为返回值'''
    PCA_eig = {}
    for i in range(len(pca)):
        PCA_eig['The {} main'.format(str(i + 1))] = [contribute[sort[len(contribute) - i - 1]], eigValues[pca[i]],
                                                     eigVectors[pca[i]]]
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')
    print('主成分:\n', PCA_eig)
    print('@@@@@@@@@@@@@@@@@@@@@@@@@@')

    # 计算方差贡献率
    tot = sum(eigValues)
    var_exp = [(i / tot) for i in sorted(eigValues, reverse=True)]
    print(var_exp)
    cum_var_exp = cumsum(var_exp)
    print(cum_var_exp)

    # 对特征值进行排序
    eigValuesIndex = argsort(eigValues)
    print('eigValuesIndex1:\n', eigValuesIndex)

    # print(eigValuesIndex)
    # 保留前K个最大的特征值
    eigValuesIndex = eigValuesIndex[:-(1000000):-1]
    print('eigValuesIndex2')
    print(eigValuesIndex)
    # print(eigValuesIndex)
    # 计算出对应的特征向量
    trueEigVectors = eigVectors[:, eigValuesIndex]
    print('trueEigVectors\n', trueEigVectors)
    # print(trueEigVectors)
    # 选择较大特征值对应的特征向量
    maxvector_eigval = trueEigVectors[:, 0]
    return maxvector_eigval
    # print(maxvector_eigval)
    # 执行PCA变换：Y=PX 得到的Y就是PCA降维后的值 数据集矩阵
    # pca_result = maxvector_eigval * Dataadjust.T
    # print(pca_result)
