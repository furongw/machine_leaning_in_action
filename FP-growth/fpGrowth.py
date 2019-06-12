####################FP树的类定义###############################
class treeNode:
    def __init__(self,nameValue,numOccur,parentNode):
        self.name = nameValue  # 名字
        self.count = numOccur  # 计数值
        self.nodeLink = None  # 链接相似的元素
        self.parent = parentNode  # 指向当前节点的父节点
        self.children = {}  # 存放节点的子节点

    def inc(self,numOccur):
        self.count += numOccur

    def disp(self, ind=1):  # 以文本形式显示树
        print(" " * ind, self.name, " ", self.count)
        for child in self.children.values():
            child.disp(ind + 1)

################FP树构建函数######################
def createTree(dataSet,minSup = 1):
    headerTable = {} #头指针表
    for trans in dataSet:#首次遍历数据集，计数
        for item in trans:
            headerTable[item] = headerTable.get(item,0) + dataSet[trans]

    for key in list(headerTable.keys()): #删除非频繁项
        if headerTable[key] < minSup:
            del(headerTable[key])
    freqItemSet = set(headerTable.keys())  # 频繁项集的键

    if len(freqItemSet) == 0:return None,None   # 如果没有元素项满足要求则退出
    for key in headerTable:
        headerTable[key] = [headerTable[key],None]

    retTree = treeNode("NUll Set",1,None)  # 初始节点
    for trans,count in dataSet.items():
        localD = {}
        for item in trans:
            if item in freqItemSet:
                localD[item] = headerTable[item][0]
        if len(localD) > 0:
            orderedItems = [v[0]for v in sorted(localD.items(),key=lambda k:k[1],reverse=True)] #根据全局频率对事务中的元素进行排序
            updateTree(orderedItems,retTree,headerTable,count) #FP树填充
    return retTree,headerTable

def updateTree(items,inTree,headerTable,count):#FP树填充(填充项，填入树，头指针表，填充的数目)
    if items[0] in inTree.children:
        inTree.children[items[0]].inc(count) #已存在该子节点，更新计数值
    else:
        inTree.children[items[0]] = treeNode(items[0],count,inTree)#不存在该子节点，新建树节点
        if headerTable[items[0]][1] == None:#更新头指针表，如果头指针表未有nodelink则创建
            headerTable[items[0]][1] = inTree.children[items[0]]
        else:  # 如果已创建，建立链表型头指针nodelink
            updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
    if len(items) > 1: #填充剩下的元素到子节点
        updateTree(items[1::],inTree.children[items[0]],headerTable,count)


def updateHeader(nodeOri,targetNode): ##更新头指针表
    # 确保节点链接指向该树中该元素的每一个实例，从头指针的nodelink开始，一直nodelink直到到达链表尾端
    while nodeOri.nodeLink != None:
        nodeOri = nodeOri.nodeLink
    nodeOri.nodeLink = targetNode


###################一个简单数据集及数据包装器###################
def loadSimpDat():
    simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
    return simpDat

##将数据集从列表转换为字典
def createInitSet(dataSet):
    retDict = {}
    for trans in dataSet:
        retDict[frozenset(trans)] = 1
    return retDict


###################发现以给定元素项结尾的所有路径的函数#################################################
##抽取条件模式基
def ascendTrees(leafNode,prefixPath):#迭代上溯FP树 leafNode表示与元素关联的头指针表，prefixPath存放上溯过程中遇到的节点
    if leafNode.parent != None:
        prefixPath.append(leafNode.name)
        ascendTrees(leafNode.parent,prefixPath)

def findPrefixPath(basePat,treeNode):#找到以元素basePat结尾的所有前缀路径，treeNode表示与元素basePat关联的头指针表
    condPats = {}
    while treeNode != None:
        prefixPath = []
        ascendTrees(treeNode,prefixPath)
        if len(prefixPath) > 1:
            condPats[frozenset(prefixPath[1:])] = treeNode.count
        treeNode = treeNode.nodeLink   # 根据头指针表访问下一个元素
    return condPats


############递归查找频繁项集的minTree函数##############
def mineTree(inTree,headerTable,minSup,preFix,freqItemList):
    '''

    :param inTree:
    :param headerTable:
    :param minSup:
    :param preFix:
    :param freqItemList:
    :return:
    首先对头指针中的元素按照其出现频率进行排序。然后将每一个频繁项添加到频繁项集列表中。接下来递归调用findPrefixPath来创建条件基
    该条件基被当成一个新数据集输送给createTree。如果书中有元素项的话，递归调用mineTree
    '''
    bigL = [v[0] for v in sorted(headerTable.items(),key=lambda  x:x[1][0])] #从头指针表的底端开始，从小到大排序
    for basePat in bigL:
        newFreqSet = preFix.copy()
        newFreqSet.add(basePat)
        freqItemList.append(newFreqSet)
        condPattBases = findPrefixPath(basePat,headerTable[basePat][1]) #前缀路径
        myCondtree,myHead = createTree(condPattBases,minSup) #FP条件树

        if myHead != None:
            myCondtree.disp(1)
            mineTree(myCondtree,myHead,minSup,newFreqSet,freqItemList)






if __name__ =='__main__':
    # rootnode = treeNode('pyramid',9,None)
    # rootnode.children['eye'] = treeNode('eye',13,None)
    # rootnode.disp()
    simpDat = loadSimpDat()
    # print(simpDat)
    iniSet = createInitSet(simpDat)
    # print(iniSet)
    myFPtree, myHeaderTab = createTree(iniSet,3)
    # myFPtree.disp()
    # print(findPrefixPath('x',myHeaderTab['x'][1]))
    # print(findPrefixPath('r', myHeaderTab['r'][1]))
    freqItems = []
    mineTree(myFPtree,myHeaderTab,3,set([]),freqItems)
    print("freqItem:", freqItems)