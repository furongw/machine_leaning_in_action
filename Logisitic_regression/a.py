import logRegres
import numpy as np
#print(logRegres.loadDataSet())
dataArr,labelMat = logRegres.loadDataSet()

#print(np.mat(labelMat))
#print(np.mat(dataArr))
weight = np.ones((3,1))
value = np.mat(dataArr)*weight
labelMat = np.mat(labelMat)
#print(labelMat)
#print(value-labelMat)
#print(np.mat(dataArr)*weight)
#weight = logRegres.gradAscent(dataArr,labelMat)
weight = logRegres.stocGradAscent1(dataArr,labelMat)
print(weight)
logRegres.plotBestFit(weight.getA())