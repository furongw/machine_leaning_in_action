import knn
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#group,labels= knn.createDataset()
#knn.classify([0, 0], group, labels, 3)

datingDataMat,datingLabels = knn.file2matrix('datingTestSet2.txt')
#print(datingDataMat)
#print(datingLabels)
'''fig = plt.figure()
ax = fig.add_subplot(111)
#ax.scatter(datingDataMat[:,1],datingDataMat[:,2])
ax.scatter(datingDataMat[:,1],datingDataMat[:,2],15*np.array(datingLabels),15*np.array(datingLabels))
plt.show()'''
normMat,ranges,minVals = knn.autoNorm(datingDataMat)
#print(normMat)

knn.datingClassTest()

