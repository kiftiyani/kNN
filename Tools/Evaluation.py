from collections import Counter


class Evaluation:
    # Usfita Kiftiyani 2017/03/16 02.44 p.m
    """class for calculating error and accuracy"""
    def __init__(self, classes):
        self.classes = classes
        self.accuracy = []
        self.errorrate = []
        self.counts = []

    # ~~function to evaluate the prediction result
    # result = [:][2] --> column 1. actual class 2. predicted class
    def evaluate(self, result, k):
        count = Counter((x[0], x[1]) for x in result)
        for act_klas in self.classes:
            for pred_klas in self.classes:
                if (act_klas, pred_klas) not in count:
                    count.update({(act_klas, pred_klas): 0})
        self.counts.append({k: count})

        tmpacc = 0
        tmperror = 0
        total = 0
        for cnt in count:
            total += count[(cnt[0], cnt[1])]
            if cnt[0] == cnt[1]:
                tmpacc += count[(cnt[0], cnt[1])]
            else:
                tmperror += count[(cnt[0], cnt[1])]

        self.accuracy.append([k, float(tmpacc)/float(total)])
        self.errorrate.append([k, tmperror])
