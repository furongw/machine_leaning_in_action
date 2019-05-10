import numpy as np
import math
import re
import random

def loadDataSet():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec = [0, 1, 0, 1, 0, 1]  # 1 is abusive, 0 not
    return postingList, classVec

#创建一个包含在所有文档中出现的不重复词的列表
def createVocabList(dataSet):
    vocabSet = set([])  # create empty set
    for document in dataSet:
        vocabSet = vocabSet | set(document)  # union of the two sets
    return list(vocabSet)

#将词汇转换成向量（是否在词汇表中）
def setOfWords2Vec(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] = 1
        else:
            print('the word {} is not in my vocabulary'.format(word))
    return returnVec


#为计算条件概率做准备
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs = len(trainMatrix)#训练集文档数
    numWords = len(trainMatrix[0])#训练集总词数
    pAbusive = sum(trainCategory)/float(numTrainDocs)#训练集中总的概率
    p0Num = np.ones(numWords); p1Num = np.ones(numWords)      #change to ones()
    p0Denom = 2; p1Denom = 2                        #change to 2.0
    for i in range(numTrainDocs):
        #如果分类为1则将该文档的所有词对应位置加入p1Num，词的总数加入p1Denom
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    #计算每个分类中每个词占总词数的比例
    p1Vect = np.log(p1Num/p1Denom)        #change to log()
    p0Vect = np.log(p0Num/p0Denom)        #change to log()
    return p0Vect,p1Vect,pAbusive

#分类函数，仍然基于贝叶斯概率
#输入待测向量，各分类向量，和分类概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify*p1Vec)+ np.log(pClass1)
    p0 = sum(vec2Classify*p0Vec)+ np.log(1-pClass1)
    if p1>p0:
        return 1
    else:
        return 0

#一个便利函数
def testingNB():
    postingList, classVec = loadDataSet()
    vocabSet = createVocabList(postingList)
    trainMat = []
    for post in postingList:
        trainMat.append(setOfWords2Vec(vocabSet,post))
    p0Vect,p1Vect,pAbusive = trainNB0(np.array(trainMat),np.array(classVec))
    testEntry = ['love', 'my', 'dalmation']
    thisDoc = np.array(setOfWords2Vec(vocabSet, testEntry))
    print (testEntry,'classified as:{} '.format(classifyNB(thisDoc,p0Vect,p1Vect,pAbusive)))
    testEntry = ['stupid', 'garbage']
    thisDoc = np.array(setOfWords2Vec(vocabSet, testEntry))
    print (testEntry,'classified as:{} '.format(classifyNB(thisDoc,p0Vect,p1Vect,pAbusive)))

#词袋模式的文字转向量：
def bagOfWords2VecMN(vocabList, inputSet):
    returnvec = [0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnvec[vocabList.index(word)]+=1
    return returnvec


###########################使用朴素贝叶斯进行交叉验证##############################
###########################例：文件解析及完整的垃圾邮件测试函数#####################
#将文本中所有词划分，去除标点和长度<2的词，返回词语列表
###这部分不知道为毛线编译不过去先跳过吧
'''
def textParse(bigstring):
    regEX = re.compile('\\W')
    text = regEX.split(bigstring)
    returntext = [tex for tex in text if len(tex)>2]
    return returntext

def spamTest():
    #全部文本集，分类集
    docList = [];classList = [];fulltex = []

    for i in range(1,26):
        file1 = 'C:\\Users\\付蓉\\PycharmProjects\\machine learning in action\\Naive_Bayes\\email\\ham\\'+str(i)+'.txt'
        p1text =textParse(open(file1).readlines())
        docList.append(p1text)
        fulltex.extend(p1text)
        classList.append(1)
        p0text = textParse(open('C:\\Users\\付蓉\\PycharmProjects\\machine learning in action\\Naive_Bayes\\email\\spam\\' + str(i) + '.txt').read())
        docList.append(p0text)
        fulltex.extend(p0text)
        classList.append(0)
    vocablist = createVocabList(docList)
    # 存储训练集和测试集
    trainingSet = range(50);testSet = []
    for i in range(10):
        randIndex = int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainingMat = [];trainClasses = []
    for docIndex in trainingSet:
        trainingMat.append(setOfWords2Vec(vocablist, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V,p1V,pSpam = trainNB0(np.array(trainingMat),np.array(trainClasses))
    errorCount = 0
    for docIndex in testSet:
        wordVector = setOfWords2Vec(vocablist, docList[docIndex])
        if classifyNB(np.array(wordVector),p0V,p1V,pSpam) != classList[docIndex]:
            errorCount +=1
    print('the error rate is {}'.format(errorCount/len(testSet)))

    # 分出训练集和测试集，进行交叉验证


    #计算准确率'''



