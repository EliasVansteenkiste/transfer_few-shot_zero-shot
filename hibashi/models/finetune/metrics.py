from hibashi.metrics.metrics import AverageAccuracy, AverageTop1ErrorRate


class AverageAccuracyFinetune(AverageAccuracy):
    def __init__(self):
        super(AverageAccuracyFinetune, self).__init__(num_classes=57)


class AverageTop1ErrorRateFinetune(AverageTop1ErrorRate):
    def __init__(self):
        super(AverageTop1ErrorRateFinetune, self).__init__(num_classes=57)
