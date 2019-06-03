import kMeans
import numpy as np
import matplotlib.pyplot as plt
dataMat = np.mat(kMeans.loadDataSet('testSet.txt'))
#print(kMeans.randCent(dataMat,2))
myCentroids, clustAssing = kMeans.kMeans(dataMat,4)
#print(myCentroids,clustAssing)
datalist = dataMat.tolist()
#print([x[0] for x in datalist])
'''plt.figure()
plt.scatter([x[0] for x in datalist],[x[1] for x in datalist])
plt.scatter([x[0] for x in myCentroids.tolist()],[x[1] for x in myCentroids.tolist()])
plt.title('kmeans')
plt.show()'''
dataMat = kMeans.loadDataSet("testSet2.txt")
centList, clusteAssment = kMeans.biKmeans(dataMat, 3)
print(centList)

