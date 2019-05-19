import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def normSample2D(x, y, var, N):
    X = np.random.normal(loc=x, scale=var, size=N)
    Y = np.random.normal(loc=y, scale=var, size=N)
    return np.dstack((X, Y))[0]


def generatePoints():
    samplesClassA = normSample2D(0, 3, 1, 40)
    samplesClassB = normSample2D(5, 0, 1, 40)
    return np.concatenate([samplesClassA, samplesClassB]), np.array([-1] * 40 + [1] * 40)


flag = 0
def iterationSignal(w):
    global flag
    flag += 1
    print('%d iteration' % flag)
    

def svmCost(weights, data, labels, l):
    predictedLabels = data.dot(weights)
    i = np.where(predictedLabels * labels < 1) # failureCaseIndices
    err = predictedLabels[i] - labels[i]
    cost = err.transpose().dot(err) + l * weights.transpose().dot(weights)
    gradient = 2 * data[i].transpose().dot(err) + 2 * l * weights
    return cost, gradient


def svm(data, labels, l):
    N, dimension = data.shape
    weights = np.random.rand(dimension + 1) # inital weights, i.e., parameters to be trained

    nX = np.ones((N, dimension + 1))
    nX[:, 1: dimension + 1] = data
    print(nX)
    res = minimize(lambda p: svmCost(p, nX, labels, l), weights, method='BFGS', jac=True, callback=iterationSignal)
    if not res.success:
        print(res.message)
    else:
        weights = res.x
    return weights


def main():
    data, labels = generatePoints()
    weights = svm(data, labels, 0.01)

    k, b = -weights[1] / weights[2], -weights[0] / weights[2]
    b1, b2 = -(weights[0] + 1) / weights[2], -(weights[0] - 1) / weights[2]
    
    plt.plot(data[:40, 0], data[:40, 1], 'rx')
    plt.plot(data[40:, 0], data[40:, 1], 'bx')
    plt.plot([-1, 6], [b - k, k * 6 + b], 'k')
    plt.plot([-1, 6], [b1 - k, k * 6 + b1], 'k--')
    plt.plot([-1, 6], [b2 - k, k * 6 + b2], 'k--')
    plt.show()
    

if __name__ == '__main__':
    main()
