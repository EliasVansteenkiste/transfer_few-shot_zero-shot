from hibashi.metrics.metrics import Fbeta, AverageAccuracy, TopKAccuracy, AverageTop1ErrorRate


class F1Jeans(Fbeta):
    def __init__(self):
        super(F1Jeans, self).__init__(19, num_classes=20, beta=1)


class F1PerfumeAndBodyMist(Fbeta):
    def __init__(self):
        super(F1PerfumeAndBodyMist, self).__init__(18, num_classes=20, beta=1)


class F1FormalShoes(Fbeta):
    def __init__(self):
        super(F1FormalShoes, self).__init__(17, num_classes=20, beta=1)


class F1Socks(Fbeta):
    def __init__(self):
        super(F1Socks, self).__init__(16, num_classes=20, beta=1)


class F1Backpacks(Fbeta):
    def __init__(self):
        super(F1Backpacks, self).__init__(15, num_classes=20, beta=1)


class F1Belts(Fbeta):
    def __init__(self):
        super(F1Belts, self).__init__(14, num_classes=20, beta=1)


class F1Briefs(Fbeta):
    def __init__(self):
        super(F1Briefs, self).__init__(13, num_classes=20, beta=1)


class F1Sandals(Fbeta):
    def __init__(self):
        super(F1Sandals, self).__init__(12, num_classes=20, beta=1)


class F1FlipFlops(Fbeta):
    def __init__(self):
        super(F1FlipFlops, self).__init__(11, num_classes=20, beta=1)


class F1Wallets(Fbeta):
    def __init__(self):
        super(F1Wallets, self).__init__(10, num_classes=20, beta=1)


class F1Sunglasses(Fbeta):
    def __init__(self):
        super(F1Sunglasses, self).__init__(9, num_classes=20, beta=1)


class F1Heels(Fbeta):
    def __init__(self):
        super(F1Heels, self).__init__(8, num_classes=20, beta=1)


class F1Handbags(Fbeta):
    def __init__(self):
        super(F1Handbags, self).__init__(7, num_classes=20, beta=1)


class F1Tops(Fbeta):
    def __init__(self):
        super(F1Tops, self).__init__(6, num_classes=20, beta=1)


class F1Kurtas(Fbeta):
    def __init__(self):
        super(F1Kurtas, self).__init__(5, num_classes=20, beta=1)


class F1SportShoes(Fbeta):
    def __init__(self):
        super(F1SportShoes, self).__init__(4, num_classes=20, beta=1)


class F1Watches(Fbeta):
    def __init__(self):
        super(F1Watches, self).__init__(3, num_classes=20, beta=1)


class F1CasualShoes(Fbeta):
    def __init__(self):
        super(F1CasualShoes, self).__init__(2, num_classes=20, beta=1)


class F1Shirts(Fbeta):
    def __init__(self):
        super(F1Shirts, self).__init__(1, num_classes=20, beta=1)


class F1Tshirts(Fbeta):
    def __init__(self):
        super(F1Tshirts, self).__init__(0, num_classes=20, beta=1)


class Top1AccuracyJeans(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyJeans, self).__init__(19, num_classes=20, top_k=1)


class Top1AccuracyPerfumeAndBodyMist(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyPerfumeAndBodyMist, self).__init__(18, num_classes=20, top_k=1)


class Top1AccuracyFormalShoes(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyFormalShoes, self).__init__(17, num_classes=20, top_k=1)


class Top1AccuracySocks(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracySocks, self).__init__(16, num_classes=20, top_k=1)


class Top1AccuracyBackpacks(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyBackpacks, self).__init__(15, num_classes=20, top_k=1)


class Top1AccuracyBelts(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyBelts, self).__init__(14, num_classes=20, top_k=1)


class Top1AccuracyBriefs(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyBriefs, self).__init__(13, num_classes=20, top_k=1)


class Top1AccuracySandals(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracySandals, self).__init__(12, num_classes=20, top_k=1)


class Top1AccuracyFlipFlops(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyFlipFlops, self).__init__(11, num_classes=20, top_k=1)


class Top1AccuracyWallets(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyWallets, self).__init__(10, num_classes=20, top_k=1)


class Top1AccuracySunglasses(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracySunglasses, self).__init__(9, num_classes=20, top_k=1)


class Top1AccuracyHeels(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyHeels, self).__init__(8, num_classes=20, top_k=1)


class Top1AccuracyHandbags(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyHandbags, self).__init__(7, num_classes=20, top_k=1)


class Top1AccuracyTops(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyTops, self).__init__(6, num_classes=20, top_k=1)


class Top1AccuracyKurtas(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyKurtas, self).__init__(5, num_classes=20, top_k=1)


class Top1AccuracySportShoes(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracySportShoes, self).__init__(4, num_classes=20, top_k=1)


class Top1AccuracyWatches(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyWatches, self).__init__(3, num_classes=20, top_k=1)


class Top1AccuracyCasualShoes(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyCasualShoes, self).__init__(2, num_classes=20, top_k=1)


class Top1AccuracyShirts(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyShirts, self).__init__(1, num_classes=20, top_k=1)


class Top1AccuracyTshirts(TopKAccuracy):
    def __init__(self):
        super(Top1AccuracyTshirts, self).__init__(0, num_classes=20, top_k=1)


class Top5AccuracyJeans(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyJeans, self).__init__(19, num_classes=20, top_k=5)


class Top5AccuracyPerfumeAndBodyMist(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyPerfumeAndBodyMist, self).__init__(18, num_classes=20, top_k=5)


class Top5AccuracyFormalShoes(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyFormalShoes, self).__init__(17, num_classes=20, top_k=5)


class Top5AccuracySocks(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracySocks, self).__init__(16, num_classes=20, top_k=5)


class Top5AccuracyBackpacks(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyBackpacks, self).__init__(15, num_classes=20, top_k=5)


class Top5AccuracyBelts(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyBelts, self).__init__(14, num_classes=20, top_k=5)


class Top5AccuracyBriefs(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyBriefs, self).__init__(13, num_classes=20, top_k=5)


class Top5AccuracySandals(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracySandals, self).__init__(12, num_classes=20, top_k=5)


class Top5AccuracyFlipFlops(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyFlipFlops, self).__init__(11, num_classes=20, top_k=5)


class Top5AccuracyWallets(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyWallets, self).__init__(10, num_classes=20, top_k=5)


class Top5AccuracySunglasses(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracySunglasses, self).__init__(9, num_classes=20, top_k=5)


class Top5AccuracyHeels(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyHeels, self).__init__(8, num_classes=20, top_k=5)


class Top5AccuracyHandbags(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyHandbags, self).__init__(7, num_classes=20, top_k=5)


class Top5AccuracyTops(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyTops, self).__init__(6, num_classes=20, top_k=5)


class Top5AccuracyKurtas(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyKurtas, self).__init__(5, num_classes=20, top_k=5)


class Top5AccuracySportShoes(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracySportShoes, self).__init__(4, num_classes=20, top_k=5)


class Top5AccuracyWatches(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyWatches, self).__init__(3, num_classes=20, top_k=5)


class Top5AccuracyCasualShoes(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyCasualShoes, self).__init__(2, num_classes=20, top_k=5)


class Top5AccuracyShirts(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyShirts, self).__init__(1, num_classes=20, top_k=5)


class Top5AccuracyTshirts(TopKAccuracy):
    def __init__(self):
        super(Top5AccuracyTshirts, self).__init__(0, num_classes=20, top_k=5)


class AverageAccuracyPretrain(AverageAccuracy):
    def __init__(self):
        super(AverageAccuracyPretrain, self).__init__(num_classes=20)


class AverageTop1ErrorRatePretrain(AverageTop1ErrorRate):
    def __init__(self):
        super(AverageTop1ErrorRatePretrain, self).__init__(num_classes=20)


