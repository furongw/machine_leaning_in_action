import bayes
import numpy as np
listOPosts,listClasses = bayes.loadDataSet()
myVocabList = bayes.createVocabList(listOPosts)
#print(myVocabList)
#print(bayes.setOfWords2Vec(myVocabList,listOPosts[0]))
trainMat = []
for postinDoc in listOPosts:
    trainMat.append(bayes.setOfWords2Vec(myVocabList,postinDoc))
#p0V, p1V, pAb = bayes.trainNB0(trainMat, listClasses)
#p0V,p1V,pAb = bayes.trainNB0(np.array(trainMat),np.array(listClasses))
#print(p1V)
#print(type(p0V))
#bayes.testingNB()
#print(bayes.textParse('zhe shi yige ceshi a !'))
bayes.spamTest()
