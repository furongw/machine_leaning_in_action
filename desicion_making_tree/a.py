import tree
from treeplotter import *

dataset,labels = tree.createDataSet()
label = labels.copy()
#classlist = [example[-1] for example in dataset]
mytree = tree.createTree(dataset,labels)
print(mytree)

#treeplotter.createPlot()
#print(treeplotter.getTreeDepth(mytree))
#createPlot(mytree)
#print(label)
print(tree.classify(mytree,label,[1,0]))