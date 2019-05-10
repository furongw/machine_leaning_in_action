import Regression
import numpy as np
import matplotlib.pyplot as plt
'''
#这部分写的很乱。。好好看看列表和矩阵吧
xArr,yArr = Regression.loadDataSet('ex0.txt')
#print(xArr)
ws = Regression.standRegres(xArr,yArr)
print(ws.T)
xMat = np.mat(xArr)
yMat = np.mat(yArr)
yHmat = xMat*ws
fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(xMat[:,1].flatten().A[0],yMat.T[:,0].flatten().A[0])
xcopy = xMat.copy()
#print(xcopy)
xcopy = sorted(xcopy.tolist(), key=lambda x: x[1])
yHat = xcopy*ws
x = [x[1] for x in xcopy]
#print(yHat.flatten().A[0])
#print(x)
y = yHat.flatten().A[0]
ax.plot(x,y)
plt.show()
print(np.corrcoef(yHmat.flatten().A[0],yMat.flatten().A[0]))'''

'''xArr,yArr = Regression.loadDataSet('ex0.txt')
#print(Regression.lwlr(xArr[0],xArr,yArr,0.1))
#print(Regression.lwlrTest(xArr,xArr,yArr,0.001))
xMat = np.mat(xArr)
sortx = sorted(xMat.tolist(),key=lambda x:x[1])
#注意这里k对拟合曲线的影响
yHat = Regression.lwlrTest(sortx,xArr,yArr,0.05)
fig = plt.figure()
plt.scatter(xMat[:,1].flatten().A[0],yArr)
plt.plot([x[1] for x in sortx],yHat,c = 'r',)
plt.show()'''

'''abX,abY = Regression.loadDataSet('abalone.txt')
ridgeweights = Regression.ridgeTest(np.mat(abX),np.mat(abY))
print(ridgeweights)
fig = plt.figure()
plt.plot(ridgeweights)
plt.show()'''

# xArr,yArr = Regression.loadDataSet('abalone.txt')
# Regression.stepWise(xArr,yArr,0.01,200)

