import adaboost
import numpy as np
#dataMat ,classlabel = adaboost.loadSimpData()
#print(dataMat)
#D = np.mat(np.ones((5,1)))/5
#print(adaboost.buildStump(dataMat,classlabel,D))
#classifierArray,aggest = adaboost.adaBoostTrainDS(dataMat,classlabel,9)
#print(aggest)

file = open('data.txt','r')
datalist = []
classlabel = []
for line in file.readlines():
    data = line.split()[:-4]
    label = int(line.split()[-1])
    datalist.append(list(map(float,data)))
    classlabel.append(label)
dataMat = np.mat(datalist)
classlabels = np.mat(classlabel)
classifierArray,aggest = adaboost.adaBoostTrainDS(dataMat,classlabel,40,100)
print(classifierArray)

adaboost.plotROC(aggest.T,classlabel)