import regTrees
import numpy as np
testMat = np.mat(np.eye((4)))
# print(testMat)
# mat0 ,mat1 = regTrees.binSplitDataSet(testMat,1,0.5)
# print(mat0)
# print(mat1)
myDat = regTrees.loadDataSet('ex00.txt')
myMat = np.mat(myDat)
# print(regTrees.createTree(myMat))
myDat1 = regTrees.loadDataSet('ex0.txt')
myMat1 = np.mat(myDat1)
print(regTrees.createTree(myMat1))
