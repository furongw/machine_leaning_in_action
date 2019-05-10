import numpy as np
from bs4 import BeautifulSoup

###############标准回归函数和数据导入函数####################
#输入文件名返回数值和标签
def loadDataSet(file):
    with open(file,encoding="utf-8") as f1:
        content = f1.readlines()
        xArre = [x.split("\t")[:-1] for x in content]
        xArr = []
        for i in xArre:  # 逐行将字符串数据转化为浮点数
            xArr.append(list(map(float,i)))
        yArr = [float(x.split("\t")[-1].strip("\n")) for x in content]
    return xArr,yArr

#计算线性回归系数矩阵：
def standRegres(xArr,yArr):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    xTx = xMat.T*xMat
    if np.linalg.det(xTx) ==0:# 求行列式判断是否可以求逆矩阵
        print('This matrix is singular,cannot do inverse')
        return
    ws = xTx.I *(xMat.T*yMat)
    return ws

##################局部加权线性回归函数###############
def lwlr(testPoint,xArr,yArr,k=1.0):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # 创建对角矩阵
    m = xMat.shape[0]
    weights = np.mat(np.eye((m)))
    for j in range(m):
        diffMat = testPoint - xMat[j,:]
        #权重值大小以指数级衰减
        weights[j,j] = np.exp(diffMat*diffMat.T/(-2*(k**2)))
    xTx = xMat.T * (weights * xMat)
    if np.linalg.det(xTx) ==0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = xTx.I * (xMat.T * (weights * yMat))
    return testPoint * ws


#k为高斯核中指数衰减的分母
def lwlrTest(testArr,xArr,yArr,k=1):
    testArr = np.mat(testArr)
    m = testArr.shape[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i],xArr,yArr,k)
    return yHat

#########################岭回归#############################


#岭回归,lambda可调
def ridgeRegres(xMat,yMat,lam=0.2):
    xTx = xMat.T*xMat
    denom = xTx + np.eye(xMat.shape[1]) * lam
    if np.linalg.det(denom) == 0:
        print('This matrix is singular, cannot do inverse')
        return
    ws = denom.I * (xMat.T*yMat)
    return ws

def ridgeTest(xArr,yArr):
    #标准化
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    yMean = np.mean(yMat,0)
    yMat = yMat - yMean
    xMeans = np.mean(xMat,0)
    xVar = np.var(xMat,0)
    xMat = (xMat - xMeans)/xVar
    #测试系数数
    numTestPts = 30
    wMat = np.zeros((numTestPts,xMat.shape[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xMat,yMat,np.exp(i-10))
        wMat[i,:] = ws.T
    return wMat

#################################前向逐步回归######################



def rssError(yArr, yHatArr):  # yArr and yHatArr both need to be arrays
    return ((yArr - yHatArr) ** 2).sum()


def regularize(xMat):  # regularize by columns
    inMat = xMat.copy()
    inMeans = np.mean(inMat, 0)  # calc mean then subtract it off
    inVar = np.var(inMat, 0)  # calc variance of Xi then divide by it
    inMat = (inMat - inMeans) / inVar
    return inMat

#eps步长，numIt迭代次数
def stepWise(xArr, yArr, eps=0.01, numIt=100):
    xMat = np.mat(xArr)
    yMat = np.mat(yArr).T
    # yMat归一化
    yMean = np.mean(yMat, 0)
    yMat = yMat - yMean
    # xMat归一化
    xMat = regularize(xMat)
    m, n = np.shape(xMat)
    # 返回迭代后的系数矩阵，每行为一次迭代，最后一行为本地迭代的最优
    returnMat = np.zeros((numIt, n))
    ws = np.zeros((n, 1))
    wsTest = ws.copy()
    wsMax = ws.copy()
    for i in range(numIt):
        print(ws.T)  # 打印上次迭代的结果
        lowestError = np.inf  # 初识化误差
        for j in range(n):  # 迭代n个特征
            for sign in [-1, 1]:  # 对每个特征加上或减去步进值对比效果
                wsTest = ws.copy()
                wsTest[j] += eps * sign
                yTest = xMat * wsTest
                rssErr = rssError(yMat.A, yTest.A)
                if rssErr < lowestError:
                    lowestError = rssErr
                    wsMax = wsTest
        ws = wsMax.copy()
        returnMat[i, :] = ws.T
    return returnMat



###后面这部分有点折腾不好了先跳过
'''########################购物信息的获取函数###########################
def scrapePage(inFile,outFile,yr,numPce,origPrc):
    from bs4 import BeautifulSoup
    fr = open(inFile); fw=open(outFile,'a') #a is append mode writing
    soup = BeautifulSoup(fr.read())
    i=1
    currentRow = soup.findAll('table', r="%d" % i)
    while(len(currentRow)!=0):
        title = currentRow[0].findAll('a')[1].text
        lwrTitle = title.lower()
        if (lwrTitle.find('new') > -1) or (lwrTitle.find('nisb') > -1):
            newFlag = 1.0
        else:
            newFlag = 0.0
        soldUnicde = currentRow[0].findAll('td')[3].findAll('span')
        if len(soldUnicde)==0:
            print ("item #%d did not sell" % i)
        else:
            soldPrice = currentRow[0].findAll('td')[4]
            priceStr = soldPrice.text
            priceStr = priceStr.replace('$','') #strips out $
            priceStr = priceStr.replace(',','') #strips out ,
            if len(soldPrice)>1:
                priceStr = priceStr.replace('Free shipping', '') #strips out Free Shipping
            print ("%s\t%d\t%s" % (priceStr,newFlag,title))
            fw.write("%d\t%d\t%d\t%f\t%s\n" % (yr,numPce,newFlag,origPrc,priceStr))
        i += 1
        currentRow = soup.findAll('table', r="%d" % i)
    fw.close()

from time import sleep
import json
import urllib3


def searchForSet(x,y,html,year,numPiece,ori_price):
    soup = BeautifulSoup(open(html,encoding="utf-8"),from_encoding="utf-8")

    for ele in soup.find_all("table",class_ = "li"):
        sold_tag = ele.find("span",class_ = "sold")
        if sold_tag != None: #只获取有已出售标志的产品

            pro_name = ele.find("div",class_ = "ttl").a.get_text() #获取产品名称
            if "new" in pro_name.lower() or "nisb" in pro_name.lower(): #new_flag 新产品标志
                new_flag = 1
            else:
                new_flag = 0
            price = float(ele.find_all("td")[4].get_text().replace("$","").replace("Free shipping","").replace(",",""))
            #只保留成套产品的数据  将产品数据存放到x,y中
            if price > ori_price*0.5:
                x.append([year,numPiece,new_flag,ori_price])
                y.append(price)


def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)

if __name__ == '__main__':
    lgX= []
    lgY = []
    setDataCollect(lgX,lgY)'''
