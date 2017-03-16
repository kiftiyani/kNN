# Usfita Kiftiyani 2017/03/15 03.05 p.m
import numpy as np
from collections import Counter
from Tools import KNNMethod as KNN
from Tools import Evaluation as ev
from sklearn.model_selection import train_test_split


# read the dataset
fl = open("Data/iris.data.txt", "r")
lines = fl.readlines()
new_lines = []
for line in lines:
    line = line.split(",")

    if len(line) < 2:
        continue

    new_lines.append(line)

classes = Counter(x[len(x)-1] for x in new_lines)
k = 5
# split the data into training and testing data
dt = np.array(new_lines).T
trainSet, testSet, trainClass, testClass = train_test_split(new_lines, dt[len(dt)-1].T, test_size=0.5, train_size=0.5)

# describe the  # k and distance method (mode default L2 Norm) 0: L2 Norm, 1: L1 Norm, 2: L Infinite Norm
tr = KNN.KNNMethods(k, mode=1)

# training the data training to get model
tr.training(trainSet)

# testing
testResult = []
for test in testSet:
    decision = tr.testing(test)

    tmp = test[:]
    tmp.append(str(decision))
    testResult.append(tmp)
# print testResult
print testResult
# training error calculation
eval = ev.Evaluation(classes)
testResult = np.asarray(testResult)
eval.evaluate(testResult[:, 4:6], k)

print eval.accuracy
print eval.errorrate