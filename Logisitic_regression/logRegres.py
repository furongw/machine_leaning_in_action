import math
import numpy as np
import matplotlib.pyplot as plt
import random

###############logistic回归梯度上升############################
'''def loadDataSet():
    fr = open('testSet.txt').readlines()
    feature = np.mat([list(map(float,line.strip("\n").split("\t")[0:-1])) for line in fr]) #(100,3)
    m,n = feature.shape
    x0 = np.ones((m,1))
    feature = np.hstack((x0,feature))
    label = np.mat([list(map(float,line.strip("\n").split("\t")[-1])) for line in fr]) #(100,1)
    return feature,label
'''
#将数据处理成数据和标签集
def loadDataSet():
    dataMat = []
    labelMat = []
    file = open('testSet.txt')
    for line in file.readlines():
        line = line.strip().split('	')
        dataMat.append([1,float(line[0]),float(line[1])])
    file.seek(0)
    for line in file.readlines():
        label = line.strip('\n').split('	')[-1]
        labelMat.append(float(label))

    return dataMat,np.mat(labelMat).T

def sigmoid(inX):
    return 1/(1+np.exp(-inX))

#梯度上升返回最佳权重
def gradAscent(dataMat,classlable):
    dataMat = np.mat(dataMat)
    classlable = np.mat(classlable)
    #学习速率
    alpha = 0.001
    #最大循环数
    maxcycle = 500
    #n为向量维数
    m,n = dataMat.shape
    #权重矩阵
    weight = np.ones((n,1))
    for i in range(maxcycle):
        value = sigmoid(dataMat*weight)
        error = classlable - value
        weight = weight + alpha*dataMat.transpose()*error
    return weight

#画出数据集和Logistic回归最佳拟合直线的函数
def plotBestFit(weights):
    import matplotlib.pyplot as plt
    dataMat,labelMat=loadDataSet()
    dataArr = np.array(dataMat)
    n = np.shape(dataArr)[0]
    xcord1 = []; ycord1 = []
    xcord2 = []; ycord2 = []
    for i in range(n):
        if int(labelMat[i])== 1:
            xcord1.append(dataArr[i,1]); ycord1.append(dataArr[i,2])
        else:
            xcord2.append(dataArr[i,1]); ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcord2, ycord2, s=30, c='green')
    x = np.arange(-3.0, 3.0, 0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x, y)
    plt.xlabel('X1'); plt.ylabel('X2')
    plt.show()

###################随机梯度上升#######################################
#在线学习算法
'''def stocGradAscent0(dataMatrix,classlabels):
    m,n = np.shape(dataMatrix)
    alpha = 0.01
    weights = np.ones(n)
    for i in range(m):
        error = classlabels[i] - sigmoid(sum(dataMatrix[i]*weights))
        weights = weights + alpha*error*dataMatrix[i]
    return weights'''

def stocGradAscent0(feature,label):
    feature = np.mat(feature)
    m,n = feature.shape
    w = np.ones((n,1))
    alpha = 0.01
    for i in range(m):
        h = sigmoid(feature[i,:] * w)
        w = w + alpha * feature[i,:].T * (label[i,:] - h)
    return w

#改进随机梯度上升算法
def stocGradAscent1(dataMatrix, classlabels, numIter = 150):
    dataMatrix = np.mat(dataMatrix)
    m,n = np.shape(dataMatrix)
    weight = np.ones((n,1))
    for j in range(numIter):
        Indexlist = list(range(m))
        for i in range(m):
            alpha = 4/(1+i+j)+0.01# alpha每次迭代时进行调整
            randIndex = np.random.randint(0,len(Indexlist))#随机选取更新
            h=sigmoid(dataMatrix[Indexlist[randIndex],:]*weight)
            error = classlabels[Indexlist[randIndex],:]-h
            weight = weight + alpha*dataMatrix[Indexlist[randIndex],:].T*error
            del (Indexlist[randIndex])
    return weight

#############示例：从氙气病预测病马的死亡率######################
def classifyVector(inX, weights):
    prob = sigmoid(inX*weights)
    if prob>0.5:
        return 1
    else:
        return 0

##测试函数大同小异，这里不再赘述













