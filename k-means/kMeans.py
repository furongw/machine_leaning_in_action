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
#返回每个值的质心和方差

def biKmeans(dataMat,k,distMeas=distEclud):#distMeas表示距离计算函数
    m,n = dataMat.shape
    clusteAssment = np.mat(np.zeros((m,2))) #存储每个点的簇分配结果:簇索引，误差
    centroid0 = np.mean(dataMat,axis = 0).tolist()[0] #初始簇质心
    centList = [centroid0] #聚类中心
    for i in range(m):
        clusteAssment[i,1] = distEclud(dataMat[i,:],centroid0) ** 2

    while len(centList) < k:
        lowestSSE = np.inf

        for i in range(len(centList)): #尝试划分每一个簇
            ptsInCurrCluster = dataMat[np.nonzero(clusteAssment[:,0] == i)[0],:]#在第i簇的点
            centroids, splitClustAss = kMeans(ptsInCurrCluster,2) #分2簇
            sseSplit = np.sum(splitClustAss[:,1],axis = 0)
            sseNotSplit = np.sum(clusteAssment[np.nonzero(clusteAssment[:,0] != i)[0],1],axis = 0)
            # print(sseSplit,sseNotSplit,sseSplit+sseNotSplit)

            if (sseSplit+sseNotSplit) < lowestSSE:
                bestCentToSplit = i #聚类误差最小的簇
                bestCentroid = centroids #2聚类的聚类中心
                bestClusterAss = splitClustAss.copy() #存储每个点的簇分配结果:簇索引，误差
                lowestSSE = sseSplit + sseNotSplit

        #更新簇分配结果
        # 还是不怎么能看得懂
        bestClusterAss[np.nonzero(bestClusterAss[:,0] == 1)[0],0] = len(centList) # 替换最佳二分簇的实际质心标签，这个新的质心的标签要注意
        bestClusterAss[np.nonzero(bestClusterAss[:,0] == 0)[0], 0] = bestCentToSplit
        centList[bestCentToSplit] = bestCentroid[0,:].tolist()[0] # 更新最佳二分簇的一个质心
        centList.append(bestCentroid[1,:].tolist()[0]) # 更新最佳二分簇的另一个质心
        clusteAssment[np.nonzero(clusteAssment[:,0] == bestCentToSplit)[0],:] = bestClusterAss # 将原簇结果用新二分簇表示
    return centList,clusteAssment
