import operator



def createDataSet():
    dataSet = [[1,1,'yes'],[1,1,'yes'],[1,0,'no'],[0,1,'no'],[0,1,'no']]
    labels = ['no surfacing','flippers']
    return dataSet, labels

import math

#计算香农熵
def clacshannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts={}
    for vector in dataSet:
        currentlabel = vector[-1]
        labelCounts[currentlabel] = labelCounts.get(currentlabel,0) + 1
    shannonEnt = 0
    for key in labelCounts:
        currentEnt = -math.log(float(labelCounts[key]/numEntries),2)
        shannonEnt += currentEnt*float(labelCounts[key]/numEntries)
    return shannonEnt

#按照给定特征划分数据集,返回分类后的数据集
def splitDataSet(dataset, axis, value):
    retDataSet = []
    for featVec in dataset:
        if featVec[axis] ==value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

#选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet):
    numfeat = len(dataSet[0])-1
    bestinfogain = 0
    bestfeatlabel = -1
    baseEnt = clacshannonEnt(dataSet)
    for axis in range(numfeat):
        testfeat = set(data[axis] for data in dataSet)
        newEnt = 0
        for value in testfeat:
            subdata = splitDataSet(dataSet,axis,value)
            sublen = len(subdata)
            subper = sublen/len(dataSet)
            newEnt += subper* clacshannonEnt(subdata)

        #信息增益是熵的减少或者是数据无序度的增加
        infogain = baseEnt - newEnt
        if (infogain>bestinfogain):
            bestinfogain = infogain
            bestfeatlabel = axis
    return bestfeatlabel

#当已经处理了所有的属性但是类标签仍然不唯一，采用多数表决的方法决定该叶子节点的分类
def majorityCnt(classlist):
    classCount = {}
    for vote in classlist:
        classCount[vote] = classCount.get(vote, 0) + 1
        sortedClassCount = sorted(classCount.iteritems(), key = operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataset, labels):
    #创建树停止条件：1.类别完全相同则停止划分
    classlist = [example[-1] for example in dataset]
    if classlist.count(classlist[0]) == len(classlist):
        return classlist[0]

    #2.遍历完所有特征时返回出现次数的类别
    if len(dataset[0]) ==1:
        return majorityCnt(classlist)

    #找出分树的最好特征，并对于该特征的取值进行递归分树
    bestfeat = chooseBestFeatureToSplit(dataset)
    bestfeatlabel = labels[bestfeat]
    tree = {bestfeatlabel:{}}
    del labels[bestfeat]
    uniques = set(example[bestfeat] for example in dataset)
    for value in uniques:
        sublabel = labels
        tree[bestfeatlabel][value] = createTree(splitDataSet(dataset,bestfeat,value), sublabel)
    return tree

#################使用决策树的分类函数#####################
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    featIndex = featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex] ==key:
            if type(secondDict[key]) ==dict:
                classlabel = classify(secondDict[key], featLabels,testVec)
            else:
                classlabel = secondDict[key]
    return classlabel


##################使用pickle模块存储决策树###################
#存储对象持久化
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename,'w')
    pickle.dump(inputTree,fw)
    fw.close()

def grabTree(filename):
    import pickle
    fr = open(filename)
    return pickle.load(fr)

