import numpy as np


class EuclideanDistances:
    # Usfita Kiftiyani 2017/03/12 11:52 a.m
    """This class contains methods to calculate distance with euclidean measures"""

    def __init__(self):
        return

    # ~~function of basic euclidean distance / L2 norm
    def norm_L2(self, obj1, obj2):
        distance = 0
        if len(obj1) == len(obj2):
            for i in range(0, len(obj1)-1):
                distance += pow(float(obj1[i]) - float(obj2[i]), 2)

            distance = np.sqrt(distance)
        return distance

    # ~~function of manhattan distance / L1 norm
    def norm_L1(self, obj1, obj2):
        distance = 0
        if len(obj1) == len(obj2):
            for i in range(0, len(obj1)-1):
                distance += abs(float(obj1[i]) - float(obj2[i]))

        return distance

    # ~~function of L infinite norm
    def norm_LInfinite(self, obj1, obj2):
        distance = 0
        if len(obj1) == len(obj2):
            for i in range(0, len(obj1)-1):
                distance = max(distance,  abs(float(obj1[i]) - float(obj2[i])))

        return distance
