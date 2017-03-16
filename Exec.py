# Usfita Kiftiyani 2017/03/15 03.05 p.m
import numpy as np
from matplotlib import pyplot as plt
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

# split the data into training and testing data
dt = np.array(new_lines).T
trainSet, testSet, trainClass, testClass = train_test_split(new_lines, dt[len(dt)-1].T, test_size=0.5, train_size=0.5,
                                                            stratify=dt[len(dt)-1].T)
classes = Counter(x for x in trainClass)
print classes
# describe the  # k and distance method (mode default L2 Norm) 0: L2 Norm, 1: L1 Norm, 2: L Infinite Norm
for mode in range(0, 3):
    train_eval = ev.Evaluation(classes)
    test_eval = ev.Evaluation(classes)
    for k in range(1, 21):
        tr = KNN.KNNMethods(k, mode)

        # training the data training to get model
        tr.training(trainSet)

        # training error calculation
        models_class = np.asarray(tr.model).T
        models_class = [models_class[len(models_class) - 1], models_class[len(models_class) - 2]]
        train_eval.evaluate(np.asarray(models_class).T, k)

        # testing
        # testResult[:][#features, actual class, predicted class]
        testResult = []
        for test in testSet:
            decision = tr.testing(test)

            tmp = test[:]
            tmp.append(str(decision))
            testResult.append(tmp)

        # testing error calculation
        testResult = np.asarray(testResult).T
        test_eval.evaluate(testResult[len(testResult)-2:len(testResult)].T, k)

    print train_eval.accuracy
    print train_eval.errorrate
    print test_eval.accuracy
    print test_eval.errorrate

    train_acc = np.asarray(train_eval.accuracy).T
    test_acc = np.asarray(test_eval.accuracy).T

    train_err = np.asarray(train_eval.errorrate).T
    test_err = np.asarray(test_eval.errorrate).T

    plt.figure(mode)
    plt.subplot(211)
    plt.plot(train_acc[0], train_acc[1], 'r', label='training data', lw=4)
    plt.plot(test_acc[0], test_acc[1], 'g', label='testing data', lw=4)
    plt.title('Learning curve - k vs accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('k')
    plt.legend(loc='upper right')

    plt.subplot(212)
    plt.plot(train_err[0], train_err[1], 'r', label='training data', lw=4)
    plt.plot(test_err[0], test_err[1], 'g', label='testing data', lw=4)
    plt.title('Learning curve - k vs error')
    plt.ylabel('Error')
    plt.xlabel('k')
    plt.legend(loc='upper right')

plt.show()
