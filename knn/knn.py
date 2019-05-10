import numpy as np
import operator
import os

def createDataset():
    group=np.array([[1,1.1],[1,1],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group, labels

#Inx,待输入向量
#dataset：数据集
#labels：数据标签
#k:紧邻个数

def classify(Inx,dataset,labels,k):
    #计算距离
    datasize=dataset.shape[0]
    copyInx=np.tile(Inx,(datasize,1))-dataset
    Inxsquare=copyInx**2
    Inxsquaresum=Inxsquare.sum(axis=1)
    d=Inxsquaresum**0.5

    #获取最小距离并统计
    minindex=d.argsort()
    classcount={}
    for i in range(k):
        kclass=labels[minindex[i]]
        classcount[kclass]=classcount.get(kclass,0)+1
    result=sorted(classcount.items(),key=operator.itemgetter(1),reverse=True)
    return (result[0][0])



###########################案例：约会网站##############################
#处理文本,转换为（向量，标签）
def file2matrix(filename):
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = np.zeros((numberOfLines,3))
    classsLabelVector = []
    index=0
    for line in arrayOLines:
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        classsLabelVector.append(int(listFromLine[-1]))
        index+=1
    return returnMat,classsLabelVector

#标准化
#返回归一化数据，极差，最小值
def autoNorm(dataSet):
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0]
    #注意要传入矩阵
    normDataSet = dataSet - np.tile(minVals,(m,1))
    normDataSet = normDataSet/np.tile(ranges,(m,1))
    return normDataSet,ranges,minVals


# 针对约会网站的测试代码(选取百分之十)
def datingClassTest():
    hoRatio = 0.10
    datingDataMat, datingLabels = file2matrix('datingTestSet2.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    m = normMat.shape[0]
    numTestVecs = int(m*hoRatio)
    errorCount = 0
    for i in range(numTestVecs):
        classifierResult = classify(normMat[i, :], normMat[numTestVecs:m,:],datingLabels[numTestVecs:m],3)
        print('the classifier came back with:{}, the real answer is:{}'.format(classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]):
            errorCount+=1
    print('the total error rate is:{}'.format(float(errorCount/float(numTestVecs))))



###################案例：手写识别数字#################################
#data prepare：将图象转换为测试向量
def img2vector(filename):
    returnVect = np.zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

def handwritingClassTest():
    #训练集
    filenames = os.listdir(r'C:\Users\付蓉\PycharmProjects\machine learning in action\knn\trainingDigits')
    numtrain = len(filenames)
    trainingMat=np.zeros((numtrain,1024))
    trainlabels=[]
    for i in range(numtrain):
        filename = (filenames[i].split('.'))[0]
        trainlabel = filename.split('_')[0]
        trainlabels.append(trainlabel)
        trainingMat[i,:] = img2vector('C:\\Users\\付蓉\PycharmProjects\\machine learning in action\\knn\\trainingDigits\\'+filenames[i])

    #测试集
    testfiles = os.listdir(r'C:\Users\付蓉\PycharmProjects\machine learning in action\knn\testDigits')
    numtest = len(testfiles)
    error = 0
    for i in range(numtest):
        testfilename = testfiles[i].split()[0]
        colabel = testfilename.split('_')[0]
        testvector = img2vector('C:\\Users\\付蓉\\PycharmProjects\\machine learning in action\\knn\\testDigits\\'+testfiles[i])
        #print('recognition result is {}, the truth is {}'.format(classify(testvector,trainingMat,trainlabels,3),colabel))
        if classify(testvector,trainingMat,trainlabels,3) !=colabel:
            error += 1
            print('{}:'.format(testfilename))
            print('recognition result is {}, the truth is {}'.format(classify(testvector,trainingMat,trainlabels,3),colabel))
    print('total test:{},error:{},true percentage:{}'.format(numtest,error,(numtest-error)/numtest))







