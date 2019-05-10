import numpy as np
import matplotlib.pyplot as plt

def loadDataSet(fileName):
    fr = open(fileName).readlines()
    data = [list(map(float,line.strip("\n").split("\t"))) for line in fr]
    return np.mat(data)

#生成叶结点
#计算dataset最后一列的均值
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])

#误差估计函数
#计算dataset最后一列的总方差
def regErr(dataSet):
    return np.var(dataSet[:,-1])* dataSet.shape[0]

#将数据集以feature=value划分为两部分
def binSplitDataSet(dataSet,feature,value):
    data0 = dataSet[np.nonzero(dataSet[:,feature] > value),:][0]
    data1 = dataSet[np.nonzero(dataSet[:,feature] <= value),:][0]
    return data0,data1

# 找到最优化分,进行了预剪枝
# ops[0] #容许的误差ops[1] #切分的最小样本数
# 该函数中的提前终止条件实际上是一种预剪枝
def chooseBestSplit(dataSet,leafType,errType,ops = (1,4)):
    tolS = ops[0]  # 容许的误差
    tolN = ops[1]  # 切分的最小样本数

    if len(set(dataSet[:,-1].T.tolist()[0])) == 1:  # 如果最后一列的数值相同则不进行划分
        return None,leafType(dataSet)  # 生成一个叶节点

    S = errType(dataSet)
    bestS = np.inf
    bestFeature = 0
    bestValue = 0

    m,n = dataSet.shape
    for feature in range(n-1):  # 对于前n-1个特征
        for value in set(dataSet[:,feature].T.tolist()[0]):  # 对于前n-1个特征中的每个值
            data0,data1 = binSplitDataSet(dataSet,feature,value)  # 对于每一个feature的value分成两个数据集

            if data0.shape[0] < tolN or data1.shape[0] < tolN: continue   # 切分的样本数小于阀值，则不划分

            newS = errType(data0) + errType(data1)  # 如果生成总方差小于旧方差则替代
            if newS < bestS:
                bestS = newS
                bestFeature = feature
                bestValue = value

    if (S - bestS) < tolS:  # 误差的减少不大，则不划分，生成叶节点
        return None,leafType(dataSet)

    data0,data1 = binSplitDataSet(dataSet,bestFeature,bestValue)
    if data0.shape[0] < tolN or data1.shape[0] < tolN:  # 如果数据集很小则不划分，生成叶节点
        return None,leafType(dataSet)
    return bestFeature,bestValue

#创建树
#ops[0] #容许的误差ops[1] #切分的最小样本数
def createTree(dataSet,leafType = regLeaf, errType = regErr, ops = (1,4)):
    feat,val = chooseBestSplit(dataSet,leafType,errType,ops)
    if feat == None: return val #创建叶子结点

    retTree = {}
    retTree["feat"] = feat
    retTree["val"] = val
    data0,data1 = binSplitDataSet(dataSet,feat,val)
    retTree["left"] = createTree(data0,leafType,errType,ops)
    retTree["right"] = createTree(data1,leafType,errType,ops)
    return retTree



