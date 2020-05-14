from hibashi.metrics.metrics import Fbeta


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
