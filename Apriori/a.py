import apriori
dataSet = apriori.loadDataSet()
C1 = apriori.createC1(dataSet)
print(set(C1))
D = []
for i in dataSet:
    D.append(set(i))
print(D)
L1,suppData0 = apriori.scanD(D,C1,0.5)