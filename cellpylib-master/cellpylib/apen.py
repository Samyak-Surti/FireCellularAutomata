import numpy as np


def apen(sequence, m=1, r=0):
    if type(sequence) is str:
        U = np.array([int(x) for x in sequence])
    elif type(sequence) is list:
        U = np.array(sequence)
    elif type(sequence) is np.ndarray:
        U = sequence
    else:
        raise Exception("unsupported sequence type: %s" % type(sequence))

    N = len(U)

    def maximum_distance(x_i, x_j):
        return max([abs(ua - va) for ua, va in zip(x_i, x_j)])

    def phi(m):
        x = [[U[j] for j in range(i, i + m - 1 + 1)] for i in range(N - m + 1)]
        C = [len([1 for x_j in x if maximum_distance(x_i, x_j) <= r]) / (N - m + 1.0) for x_i in x]
        return (1 / (N - m + 1.0)) * sum(np.log(C))

    return abs(phi(m + 1) - phi(m))
