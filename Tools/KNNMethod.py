import operator
from collections import Counter
from EuclideanDistances import EuclideanDistances


class KNNMethods:
    # Usfita Kiftiyani 2017/03/12 11:55 a.m
    """class to determine KNN Method"""

    def __init__(self, k, mode):
        self.model = []
        self.euc = EuclideanDistances()
        self.mode = mode
        self.k = k

    # ~~function to train the training set and save the model
    # model = [:][#features, new class, actual class]
    def training(self, trainingSet):
        for train1 in trainingSet:
            dists = []

            for train2 in trainingSet:
                dist = self.distancemode(train1, train2)

                tmp = train1[:]
                tmp.extend(train2)
                tmp.append(str(dist))

                dists.append(tmp)

            dists.sort(key=lambda x: x[2*len(train1)])

            result = Counter(x[len(x)-2] for x in dists[0:self.k])
            decision = max(result.iteritems(), key=operator.itemgetter(1))[0]

            tmp = train1[0:len(train1)-1]
            tmp.append(str(decision))
            tmp.append(train1[len(train1)-1])

            self.model.append(tmp)

    # ~~function to testing data
    def testing(self, testData):
        dists = []

        for mod in self.model:
            dist = self.distancemode(testData, mod[0:len(mod)-1])

            tmp = testData[:]
            tmp.extend(mod[0:len(mod)-1])
            tmp.append(str(dist))

            dists.append(tmp)

        dists.sort(key=lambda x: x[2 * len(testData)])

        result = Counter(x[len(x) - 2] for x in dists[0:self.k])
        decision = max(result.iteritems(), key=operator.itemgetter(1))[0]
        return decision

    # ~~function for choosing distance measure
    def distancemode(self, obj1, obj2):
        if self.mode == 1:
            return self.euc.norm_L1(obj1, obj2)
        elif self.mode == 2:
            return self.euc.norm_LInfinite(obj1, obj2)
        else:
            return self.euc.norm_L2(obj1, obj2)
