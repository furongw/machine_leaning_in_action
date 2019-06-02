import numpy as np
#####10.1 k-means聚类算法####################################################
##加载数据集
def loadDataSet(fileName):
    dataMat = []
    with open(fileName) as f1:
        content = f1.readlines()
        for line in content:
            line = list(map(float,line.strip("\n").split("\t")))
            dataMat.append(line)
    f1.close()
    return np.mat(dataMat)

#计算欧式距离
def distEclud(vecA,vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB,2)))

##为给定数据集构建k个随机质心
def randCent(dataSet,k):
    m,n = dataSet.shape
    centroids = np.mat(np.zeros((k,n)))

    for i in range(n):
        minValue = np.min(dataSet[:,i])
        maxValue = np.max(dataSet[:,i])
        Range = maxValue - minValue
        centroids[:,i] = minValue + Range * np.random.rand(k,1)
    return centroids

def kMeans(dataMat,k,distMeas=distEclud,createCent = randCent): #distMeas表示距离计算函数，createCent表示创建初始质心函数
    m,n = dataMat.shape
    clusteAssment = np.mat(np.zeros((m,2))) #存储每个点的簇分配结果:簇索引，误差
    centroids = createCent(dataMat,k) #初始质心
    clusterChanged = True #标志变量
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            #初始化
            minDist = np.inf
            minIndex = -1
            #寻找最近质心
            for j in range(k):
                dist = distMeas(dataMat[i,:],centroids[j,:])
                if dist<minDist:
                    minDist=dist
                    minIndex = j
            if clusteAssment[i,0]!=minIndex:
                clusterChanged = True
            clusteAssment[i,:]=minIndex,minDist**2
        #更新质心的位置
        for c in range(k):
            index = dataMat[np.nonzero(clusteAssment[:,0]==c)[0]]
            centroids[c,:] = np.mean(index,axis=0)
    return centroids,clusteAssment

