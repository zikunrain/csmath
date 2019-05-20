import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-6
MAXITER = 1000

def gauss(parameters, x, isNoise = False):
    a, b, c, offset = parameters
    n = np.random.normal(size=len(x), scale=0.05) if isNoise else np.zeros((len(x),))
    return a * np.exp(- (x - b) ** 2 / (2 * c ** 2)) + offset + n


def gaussJacobian(parameters, x, v):
    a, b, c, offset = parameters
    f = a * np.exp(- (x - b) ** 2 / (2 * c ** 2))
    J = np.zeros(shape=(4,) + x.shape)
    J[0] = 1. / a * f
    J[1] = f * (x - b) / c ** 2
    J[2] = f * (x - b) ** 2 / c ** 3
    J[3] = 1
    return f + (offset - v), J


def LM(fn, parameters, args, tau = 1e-4):
    p = parameters
    I = np.eye(len(parameters))
    f, J = fn(p, *args)
    JJ = np.inner(J, J)
    JF = np.inner(J, f)
    k, nu = 0, 2
    lam = tau * np.max(np.diag(JJ))
    stop = np.linalg.norm(JF, np.Inf) < EPS

    while not stop and k < MAXITER:
        k += 1
        pDelta = np.linalg.solve(JJ + lam * I, -JF)
        if np.linalg.norm(pDelta) < EPS * (np.linalg.norm(p) + EPS):
            break
        pnew = p + pDelta
        fnew, Jnew = fn(pnew, *args)
        rho = (np.linalg.norm(f) ** 2 - np.linalg.norm(fnew) ** 2) / np.inner(pDelta, lam * pDelta - JF)
        if rho > 0:
            # update
            p = pnew
            f = fnew
            J = Jnew
            JJ = np.inner(Jnew, Jnew)
            JF = np.inner(Jnew, fnew)
            # update end

            if np.linalg.norm(JF, np.Inf) < EPS:
                break
            lam = lam * np.max([1. / 3, 1. - (2. * rho - 1.) ** 3])
            nu = 2.
        else:
            lam = lam * nu
            nu = 2 * nu
        print('\nIter %d: |f|=%f, lam=%f, rho=%f' % (k, np.linalg.norm(f), lam, rho))
        print('updated p %s' % str(pnew.tolist()))
    else:
        print('max iteration reached')
    return p


def main():
    parameters, initialParameters = [1, 0.1, 1, 0.5], [4, 1, 2, 0.3]
    x = np.linspace(-4, 4, 200)
    fig = plt.figure()

    yTrue = gauss(parameters, x)
    green_ele, = plt.plot(x, yTrue, 'g')

    yNoise = gauss(parameters, x, isNoise=True)
    blue_ele, = plt.plot(x, yNoise, 'bx')

    pn = LM(gaussJacobian, initialParameters, (x, yNoise))
    yFIt = gauss(pn, x)
    red_ele, = plt.plot(x, yFIt, 'r')

    fig.legend((green_ele, blue_ele, red_ele), ('gauss', 'sample', 'fit'), 'upper right')

    plt.show()

if __name__ == '__main__':
    main()
