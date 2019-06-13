import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    with open(fileName) as f1:
        content = f1.readlines()
        lines = [list(map(float,line.strip("\n").split("\t"))) for line in content]
    return np.mat(lines)

def pca(dataMat,topNfeat=9999999):
    '''

    :param dataMat:数据矩阵
    :param topNfeat: 保留特征值数量
    :return:
    '''
    meanVals = np.mean(dataMat,axis=0)  # 求每一列的均值
    meanRemoved = dataMat-meanVals  # 每一行去对应均值
    covMat = np.cov(meanRemoved,rowvar=0)  # 计算协方差，rowvar=0意味着每一列是一个变量
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))  # 计算协方差矩阵的特征值和特征向量
    eigValInd = np.argsort(eigVals)  # 返回特征值从小到大排序的标签
    eigValInd = eigValInd[:-(topNfeat+1):-1]  # 选取前topNfeat个
    redEigVects = eigVects[:,eigValInd]
    lowDDataMat = meanRemoved*redEigVects  # 将数据转换到新空间
    reconMat = (lowDDataMat*redEigVects.T) +meanVals
    return lowDDataMat,reconMat



if __name__=='__main__':
    dataMat = loadDataSet("testSet.txt")
    lowData, reconMat = pca(dataMat, 1)
    print(lowData.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker="X", s=90)
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker="o", s=50, c="red")
    plt.show()