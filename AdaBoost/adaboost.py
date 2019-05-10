import numpy as np
import matplotlib.pyplot as plt

def loadSimpData():
    dataMat = np.mat([[1.0,2.1],[2.0,1.1],[1.3,1.0],[1.0,1.0],[2.0,1.0]])
    classlabels = [1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classlabels

######单层决策树生成函数##########
def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):#通过阈值比较对数据进行分类
    # dimen表示数据的第dimen个维度，threshval表示阈值，threshIneq表示不等号
    returnlabel = np.ones((dataMatrix.shape[0],1))
    if threshIneq == 'lt':#小于
        returnlabel[dataMatrix[:,dimen]<=threshVal] = -1
    else:
        returnlabel[dataMatrix[:,dimen] >threshVal] = -1
    return returnlabel#返回分类后标签列向量

#classlabel行、D列,step 步径
def buildStump(dataArr,classLabels,D,step = 10):#遍历stumpClassify的所有可能输入值，找到最佳单层决策树
    #D表示数据的权重向量
    dataMat = np.mat(dataArr)
    classLabels = np.mat(classLabels)
    classLabels = classLabels.T
    dimennum = dataMat.shape[1]
    minerr = np.inf
    bestStump = {}#记录最优决策树的相关信息
    for dimen in range(dimennum):#每一列特征
        dimenmin = dataMat[:,dimen].min()
        dimenmax = dataMat[:,dimen].max()
        stepsize = (dimenmax-dimenmin)/step
        for i in range(-1,int(step)+1):#对该维的值上进行遍历
            threshVal = dimenmin + i*stepsize
            for threshIneq in ['lt','gt']:#小于或大于分类
                predictedlabel = stumpClassify(dataMat,dimen,threshVal,threshIneq)
                errArr = np.ones((dataMat.shape[0],1))
                errArr[predictedlabel==classLabels]=0
                weightederr = D.T*errArr
                #print('split:dim:{},thresh:{},threshineq:{},weightederror:{}'.format(dimen,threshVal,threshIneq,weightederr))
                if weightederr < minerr:
                    minerr = weightederr
                    bestClass = predictedlabel.copy()
                    bestStump['dimen']=dimen
                    bestStump['threshVal']=threshVal
                    bestStump['threshIneq']=threshIneq
    return bestStump,minerr,bestClass#返回最好的分树桩（包括最佳维度，分类值，不等号），最小误差，最好的分类结果
#bestclass列,minArr1*1二维列表格式注意！

#classlabels行step步径
def adaBoostTrainDS(dataArr,classlabels,numIt = 40,step = 10):
    '''伪代码：
    对每次迭代：
        利用buildStump函数找到最佳的单层决策树
        将最佳单层决策树加入到单层决策树数组
        计算alpha
        计算新的权重向量D
        更新累计类别估计值
        如果错误率=0，退出循环'''
    classlabels = np.mat(classlabels)
    weakClassArr = []
    D = np.mat(np.ones((dataArr.shape[0],1)))/dataArr.shape[0]#初始化权重矩阵
    aggEst = np.zeros((dataArr.shape[0],1))#每个数据的类别估计累计值
    for i in range(numIt):
        bestStump,minerr,bestclass = buildStump(dataArr,classlabels,D,step)
        minerr = float(minerr)
        alpha = float(0.5*np.log((1-minerr)/max(minerr,1e-6)))#防止除零
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        expo = np.multiply(-1.0*alpha*classlabels.T,bestclass)
        D = np.multiply(D,np.exp(expo))
        D = D/D.sum()#更新数据权重D
        aggEst += alpha*bestclass
        errorsum = np.multiply(np.sign(aggEst)!=classlabels.T,np.ones((dataArr.shape[0],1)))#预测错误矩阵
        errorrate = np.sum(errorsum)/dataArr.shape[0]
        print('D:{},classEst:{},aggEst:{},errorrate:{}'.format(D, bestclass, aggEst, errorrate))
        if errorrate==0:
            break
    return weakClassArr,aggEst


##############Adaboost分类函数############
def adaClassify(data,classfierArray): #adaboost分类函数
    #data,classfierArray分别表示待分类数据和多个弱分类器组成的数组
    dataMat = np.mat(data)
    m,n = dataMat.shape
    preLabel = np.mat(np.zeros((m,1)))

    for i in classfierArray:
        preScore = stumpClassify(dataMat,i['dim'],i['thresh'],i['ineqal'])
        preLabel += i['alpha'] * preScore
        # print("preLabel:",preLabel)
    return np.sign(preLabel)


####################ROC曲线的绘制及AUC计算函数#######################
#ROC曲线x轴代表false positive rate(FPR = FP/(FP+FN)) y轴表示true positive rate(TPR = TP/(TP+FN))
 #preScore,labels分别表示分类器的预测强度和实际标签
def plotROC(predStrengths, classlabels):
    cur = (1,1)#光标位置
    ysum = 0#用于计算AUC的值
    pos = np.sum(np.array(classlabels)==1)#正例样本个数
    ystep = 1/float(pos)
    xstep = 1/float(len(classlabels)-pos)
    sortindex = predStrengths.argsort().tolist()[0]#从小到大排序，返回索引
    fig = plt.figure()
    fig.clf()
    for i in sortindex:
        if classlabels[i] ==1:#标签为1，在y轴倒退一个步长
            delx = 0
            dely = ystep
        else:#标签为-1，在x轴倒退一个步长
            delx = xstep
            dely = 0
            ysum +=cur[1]
        plt.plot([cur[0],cur[0]-delx],[cur[1],cur[1]-dely], c='b')
        cur = (cur[0]-delx,cur[1]-dely)
    plt.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.show()
    # 计算AUC需要对多个小矩形的面积进行累加，这些小矩形的宽度都是xStep，
    # 因此可对所有矩形的高度进行累加，然后再乘以xStep得到其总面积
    print('The AUC is {}'.format(ysum*xstep))









