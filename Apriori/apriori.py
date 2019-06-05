def loadDataSet():
    return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]

# 对C1中每个项构建一个不变集合
def createC1(dataSet):
    C1 = []
    for transaction in dataSet:
        for item in transaction:
            if not [item] in C1:
                C1.append([item])
    C1.sort()
    return map(frozenset, C1)

##从C1生成L1(大小为1的所有频繁集)
def scanD(D,Ck,minSupport):#D为数据集，Ck为候选集，minSupport为最小支持度
    ssCnt = {} #存放候选项的出现次数
    numItems = len(D) #数据集大小
    retList = [] #存放频繁项
    supportData = {} #存放频繁项集支持度

    for tid in D:
        for can in Ck:
            if can.issubset(tid):
                if can not in ssCnt.keys():
                    ssCnt[can] = 1
                else:
                    ssCnt[can] += 1

    for key in ssCnt.keys():
        support = float(ssCnt[key])/float(numItems)
        if support >= minSupport:
            retList.append(key)
        supportData[key] = support
#返回频繁项集和所对应的支持度
    return retList,supportData