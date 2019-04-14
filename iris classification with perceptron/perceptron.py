from sys import path
path.append('..')

from random import random
import util.vectors as ve
from util.dataprep import get_input_vector


class Perceptron:

    def __init__(self, act_fun, learning_rate, dim):
        self.act_fun = act_fun
        self.learning_rate = learning_rate
        self.dim = dim + 1
        self.W = [random() for i in range(dim)]

    def learn(self, accuracy, training, max_epochs=1000):
        correct, acc, epochs = 0, 0, 0
        traceback = []

        while acc < accuracy and epochs <= max_epochs:
            correct = 0
            for observation in training:
                exp_output = observation[-1]
                X = get_input_vector(observation)
                output = self.classify(X)
                if output == exp_output:
                    correct += 1
                self.delta_rule(X, exp_output, output)
                # print(observation, output)
                # normalize(self.W)
            acc = correct / len(training)
            traceback.append(acc)
            epochs += 1

        return traceback

    def classify(self, X):
        net = ve.dot_product(self.W, X)
        return self.act_fun(net)

    def delta_rule(self, X, expected_output, real_output):
        coef = (expected_output - real_output) * self.learning_rate
        T = ve.multiply_vect_by_num(X, coef)
        self.W = ve.vectors_sum(self.W, T)
