import random
import numpy

####################简化版SMO算法，在数据集上遍历每一个alpha，然后在剩下的alpha集合中随机选择另一个alpha########################
######SMO算法中的辅助函数########################

def loadDataSet(fileName):
    dataMat = []; labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = line.strip().split('\t')
        dataMat.append([float(lineArr[0]), float(lineArr[1])])
        labelMat.append(float(lineArr[2]))
    return dataMat,labelMat

##在0~m间选择一个不同于i的数##
def selectJrand(i,m):
    j = i
    while (j==i):
        j = random.randint(0,m)
    return j

#调整超出L~H范围的alpha值
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L>aj:
        aj = L
    return aj

#简化版SMO算法
#def smoSimple(dataMatIn, classLabels, C, toler, maxIter):

