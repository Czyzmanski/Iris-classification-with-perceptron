def accuracy(TP, TN, FP, FN):
    return round((TP + TN) / (TP + TN + FP + FN), 2)


def precision(TP, FP):
    try:
        return round(TP / (TP + FP), 2)
    except ZeroDivisionError:
        return 'NA'


def recall(TP, FN):
    try:
        return round(TP / (TP + FN), 2)
    except ZeroDivisionError:
        return 'NA'
