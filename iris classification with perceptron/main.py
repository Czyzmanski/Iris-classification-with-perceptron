from sys import path
path.append('..')

import util.dataprep as dp
import util.classeval as ce
import util.actfun as af
from perceptron import Perceptron


def classify_testing_data():
    TP, FP, FN, TN = 0, 0, 0, 0
    for observation in testing:
        X = dp.get_input_vector(observation)
        decision = perceptron.classify(X)
        exp_decision = observation[-1]

        if exp_decision == 1 and decision == 1:
            TP += 1
        elif exp_decision == 1 and decision == 0:
            FN += 1
        elif exp_decision == 0 and decision == 1:
            FP += 1
        else:
            TN += 1

        print(observation, decision)

    print('Accuracy = {}'.format(ce.accuracy(TP, TN, FP, FN)))
    print('Precision = {}'.format(ce.precision(TP, FP)))
    print('Recall = {}'.format(ce.recall(TP, FN)))


training_file = input('Please input path to training file:\n').strip()
testing_file = input('Please input path to testing file:\n').strip()

with open(training_file) as ftrain, open(testing_file) as ftest:
    training = dp.prepare_data(ftrain)
    testing = dp.prepare_data(ftest)

    labels = dp.prepare_labels(training)
    dp.label_data(training, labels[1])
    dp.label_data(testing, labels[1])

    print(labels)

    print('Training data:')
    for t in training:
        print(t)
    print('Testing data:')
    for t in testing:
        print(t)

    alfa = float(input('Please input learning rate:\n'))
    perceptron = Perceptron(af.step, alfa, len(training[0]) - 1)
    print(perceptron.learn(0.99, training))

    classify_testing_data()

    while True:
        observation = dp.prompt_vector()
        decision = perceptron.classify(observation)
        decision_class = labels[decision]
        print('Decision output: {}\nDecision class: {}'.format(decision, decision_class))
