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
    return list(map(frozenset, C1))

##从C1生成L1(大小为1的所有频繁集)
def scanD(D,Ck,minSupport):#D为数据集的集合，Ck为候选集，minSupport为最小支持度
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

def aprioriGen(Lk,k):
    ##Lk为频繁项，k为项集元素个数
    #如输入{0},{1},{2}返回{0，1}，{1，2}，{0，2}
    retlist = []
    for i in range(len(Lk)-1):
        for j in range(i+1,len(Lk)):  # 前k-2个项相同时，将两个集合合并
            L1 = list(Lk[i])[:k-2]
            L1.sort()
            L2 = list(Lk[j])[:k-2]
            L2.sort()
            if L1 == L2:
                retlist.append((Lk[i] | Lk[j]))
    return retlist

# 产生最小支持度为minSupport的频繁项集
def apriori(dataSet,minSupport = 0.5):
    C1 = createC1(dataSet)
    D =list(map(set,dataSet))

    L1,supportData = scanD(D,C1,minSupport)  # 返回频繁项集和对应的支持度
    L= [L1]
    k=2
    while len(L[k-2]) > 0:     # 一直运行直到无法再合并
        Ck = aprioriGen(L[k-2],k)  # 产生包含k个元素的项集
        Lk, supK = scanD(D, Ck, minSupport)  # 返回包含k个元素的频繁项集和支持度
        supportData.update(supK)  # 将新生成的频繁项集和指出度加入列表中
        L.append(Lk)
        k += 1

    return L,supportData

########################从频繁项集中挖掘关联规则########################












if  __name__ == '__main__':
    dataSet = loadDataSet()
    C1 = createC1(dataSet)
    # print(set(C1))
    D = list(map(set, dataSet))
    L1, suppData0 = scanD(D, C1, 0.5)
    L,suppData = apriori(dataSet,0.7)
    print(L)