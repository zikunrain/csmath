import numpy as np
import matplotlib.pyplot as plt

EPS = 1e-6
MAXITER = 100

def gauss(pars, x, v0 = 0, isNoise = False):
    a, b, c, offset = pars[:4]
    n = np.random.normal(size=len(x), scale=0.05) if isNoise else np.zeros((len(x),))
    return a * np.exp(- (x - b) ** 2 / (2 * c ** 2)) + offset - v0 + n


def gauss_jacobian(pars, x, v0 = 0):
    a, b, c, offset = pars[:4]
    f = a * np.exp(- (x - b) ** 2 / (2 * c ** 2))
    J = np.zeros(shape=(4,) + x.shape)
    J[0] = 1. / a * f
    J[1] = f * (x - b) / c ** 2
    J[2] = f * (x - b) ** 2 / c ** 3
    J[3] = 1
    return f + (offset - v0), J


def LM(fn, pars, args, tau = 1e-4):
    p, I = pars, np.eye(len(pars))
    f, J = fn(p, *args)
    A, g,  = np.inner(J, J), np.inner(J, f)
    k, nu, mu = 0, 2, tau * np.max(np.diag(A))
    stop = np.linalg.norm(g, np.Inf) < EPS

    while not stop and k < MAXITER:

        k += 1
        d = np.linalg.solve(A + mu * I, -g)
        if np.linalg.norm(d) < EPS * (np.linalg.norm(p) + EPS):
            break
        pnew = p + d
        fnew, Jnew = fn(pnew, *args)
        rho = (np.linalg.norm(f) ** 2 - np.linalg.norm(fnew) ** 2) / np.inner(d, mu * d - g)
        if rho > 0:
            p, A, g, f, J = pnew, np.inner(Jnew, Jnew), np.inner(Jnew, fnew), fnew, Jnew
            if np.linalg.norm(g, np.Inf) < EPS:
                break
            mu = mu * np.max([1. / 3, 1. - (2. * rho - 1.) ** 3])
            nu = 2.
        else:
            mu = mu * nu
            nu = 2 * nu
        print('Iter %d: |f|=%f, mu=%f, rho=%f' % (k, np.linalg.norm(f), mu, rho))
    else:
        print('max iteration reached')
    return p


def main():
    pars = [1, 0.1, 1, 0.5]
    ipars = [2, 0.5, 2, 0.5]
    x = np.linspace(-3, 3, 300)

    yv = gauss(pars, x)
    plt.plot(x, yv, 'g')

    y = gauss(pars, x, isNoise=True)
    plt.plot(x, y, 'bx')

    pn = LM(gauss_jacobian, ipars, (x, y))
    print(pn)
    yn = gauss(pn, x)
    plt.plot(x, yn, 'r')
    plt.show()

if __name__ == '__main__':
    main()
