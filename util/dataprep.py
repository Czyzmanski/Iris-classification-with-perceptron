def prepare_data(f):
    data = [line.strip().split(',') for line in f]
    for observation in data:
        for i in range(len(observation) - 1):
            observation[i] = float(observation[i])
    return data


def label_data(data, positive_class):
    for observation in data:
        class_name = observation[-1]
        class_label = 1 if class_name == positive_class else 0
        observation.append(class_label)


def prepare_labels(training):
    positive_class = training[0][-1]
    labels = {1: positive_class}
    i = 1
    while i < len(training) and training[i][-1] == positive_class:
        i += 1
    labels[0] = training[i][-1]
    return labels


def get_input_vector(observation):
    X = observation[0:-2]
    X.append(-1)
    return X


def prompt_vector():
    return [float(a) for a in input('Please input vector separating attributes by white spaces:\n').strip().split()]
