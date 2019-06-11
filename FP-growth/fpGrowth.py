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





if __name__ =='__main__':
    # rootnode = treeNode('pyramid',9,None)
    # rootnode.children['eye'] = treeNode('eye',13,None)
    # rootnode.disp()
    simpDat = loadSimpDat()
    print(simpDat)
    iniSet = createInitSet(simpDat)
    print(iniSet)
    myFPtree, myHeaderTab = createTree(iniSet,3)
    myFPtree.disp()

