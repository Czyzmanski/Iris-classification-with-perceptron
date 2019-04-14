def dot_product(v, u):
    return sum(a*b for a, b in zip(v, u))


def vector_len(v):
    return sum(a**2 for a in v) ** (1/2)


def normalize(v):
    L = vector_len(v)
    for i in range(len(v) - 1):
        v[i] = v[i] / L


def vectors_sum(v, u):
    return [a+b for a, b in zip(v, u)]


def multiply_vect_by_num(v, num):
    return [num*a for a in v]
