from math import e


def step(net):
    return 1 if net >= 0 else 0


def uni_sigmoid(net, alfa=1):
    return 1 / (1 + pow(e, -net * alfa))
