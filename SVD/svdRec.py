import numpy as np

def loadExData():
    return[[0, 0, 0, 2, 2],
           [0, 0, 0, 3, 3],
           [0, 0, 0, 1, 1],
           [1, 1, 1, 0, 0],
           [2, 2, 2, 0, 0],
           [5, 5, 5, 0, 0],
           [1, 1, 1, 0, 0]]

###############三种相似度计算方法##########
#欧式距离
def euclidSum(inA,inB):
    return  1.0/(1.0 +np.linalg.norm(inA-inB))

# 皮尔逊相关系数
def pearsSim(inA,inB):
    if len(inA)<3:return 1
    return 0.5+0.5*np.corrcoef(inA,inB,rowvar = 0)[0][1]  # 把取值归一化到0到1之间

# 余弦相似度
def cosSim(inA,inB):
    num = float(inA.T*inB)
    denom = np.linalg.norm(inA)* np.linalg.norm(inB)
    return 0.5+0.5*(num/denom)


##############基于物品相似度的推荐引擎##################33
# 计算在给定相似度计算方法的条件下，用户对物品的估计评分值
def standEst(dataMat, user, simMeas, item):
    n = np.shape(dataMat)[1]
    simTotal = 0.0
    ratSimTotal = 0.0
    for j in range(n):
        userRating = dataMat[user, j]
        # 如果某个物品的评分值为0，则跳过这个物品
        if userRating == 0:
            continue
        # 寻找两个用户都评级的物品
        # 变量overLap给出的是两个物品当中已经被评分的那个元素的索引ID,.A将矩阵转为array
        overLap = np.nonzero(np.logical_and(dataMat[:, item].A > 0, dataMat[:, j].A > 0))[0]
        # 如果相似度为0，则两者没有任何重合元素，终止本次循环
        if len(overLap) == 0:
            similarity = 0
        # 如果存在重合的物品，则基于这些重合物重新计算相似度
        else:
            similarity = simMeas(dataMat[overLap, item], dataMat[overLap, j])
        print('the %d and %d similarity is: %f' % (item, j, similarity))
        # 相似度会不断累加，每次计算时还考虑相似度和当前用户评分的乘积
        # similarity 评了两样物品的用户对于两样物品评分的相似度   userRating 用户评分
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0:
        return 0
    # 通过除以所有的评分和，对上述相似度评分的乘积进行归一化，使得最后评分在0~5之间，这些评分用来对预测值进行排序
    else:
        return ratSimTotal / simTotal

#推荐引擎
def recommend(dataMat, user, N=3, simMeas=cosSim, estMethod=standEst):
    # 寻找未评级的物品
    # 对给定的用户建立一个未评分的物品列表
    unratedItems = np.nonzero(dataMat[user, :].A == 0)[1]
    # 如果不存在未评分物品，那么就退出函数
    if len(unratedItems) == 0:
        return ('you rated everything')
    # 物品的编号和评分值
    itemScores = []
    for item in unratedItems:
        estimatedScore = estMethod(dataMat, user, simMeas, item)
        # 寻找前N个未评级的物品，调用standEst()来产生该物品的预测得分，该物品的编号和估计值会放在一个元素列表itemScores中
        itemScores.append((item, estimatedScore))
    # 返回元素列表，第一个就是最大值
    return sorted(itemScores, key=lambda jj: jj[1], reverse=True)[: N]

##################基于SVD提高推荐的效果##################
def loadExData2():
    return[[0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 5],
           [0, 0, 0, 3, 0, 4, 0, 0, 0, 0, 3],
           [0, 0, 0, 0, 4, 0, 0, 1, 0, 4, 0],
           [3, 3, 4, 0, 0, 0, 0, 2, 2, 0, 0],
           [5, 4, 5, 0, 0, 0, 0, 5, 5, 0, 0],
           [0, 0, 0, 0, 5, 0, 1, 0, 0, 5, 0],
           [4, 3, 4, 0, 0, 0, 0, 5, 5, 0, 1],
           [0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4],
           [0, 0, 0, 2, 0, 2, 5, 0, 0, 1, 2],
           [0, 0, 0, 0, 5, 0, 0, 0, 0, 4, 0],
           [1, 0, 0, 0, 0, 0, 0, 1, 2, 0, 0]]

# 实际矩阵要稀疏得多
# 注意要合理选择num使得总能量不低于约原数据的能量的90%
def svdEst(dataMat,user,simMeas,item,num):
    n = dataMat.shape[1]
    simTotal = 0
    ratSimTotal = 0
    U, Sigma, VT = np.linalg.svd(dataMat)
    Sig4 = np.mat(np.eye(num)*Sigma[:num])  # 建立奇异值的对角矩阵
    xformedItems = dataMat.T * U[:,:num] * Sig4.I  #构建转换后的物品
    for j in range(n):
        userRating = dataMat[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T,xformedItems[j,:].T)
        print ('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal



if __name__=='__main__':
    Data = loadExData()
    U,Sigma,VT = np.linalg.svd(Data)
    # print(Sigma)
    Sig3 = np.mat([[Sigma[0],0,0],[0,Sigma[1],0],[0,0,Sigma[3]]])
    # print(U[:,:3]*Sig3*VT[:3,:])
    myMat = np.mat(Data)
    print(euclidSum(myMat[:,0],myMat[:,0]))  # 注意以上相似度计算都是假设数据采用了列向量方式进行表示
