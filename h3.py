import math
import numpy as np
import matplotlib.pyplot as plt
import operator
import copy

def generate_2d_gaussian(N, meanX, meanY, var):
    x = np.random.normal(loc=meanX, scale=var, size=N)
    y = np.random.normal(loc=meanY, scale=var, size=N)
    return x, y

def generate_data():
    x0, y0 = generate_2d_gaussian(300, 0, 0, 1.5)
    x1, y1 = generate_2d_gaussian(200, 7, 2, 1)
    x2, y2 = generate_2d_gaussian(200, 3, 5, 1.5)
    data = np.dstack((np.concatenate((x0, x1, x2)), np.concatenate((y0, y1, y2))))[0]
    return data

def initialize_params(pts, K):
    N = len(pts)
    dim = len(pts[0])
    ci = np.random.randint(N, size=K)

    miu = pts[ci]
    pi = np.zeros((K,))
    sigma = np.zeros((K, dim, dim))

    dist = np.tile(np.sum(pts * pts, 1), (K, 1)).transpose() + \
        np.tile(np.sum(miu * miu, 1), (N, 1)) - \
        2 * pts.dot(miu.transpose())
    m = np.argmin(dist, axis=1)

    for k in range(K):
        pk = pts[np.where(m == k)]
        pi[k] = len(pk) / N
        sigma[k] = np.cov(pk.transpose())

    return miu, pi, sigma

def estimate(pts, K, miu, pi, sigma):
    N = len(pts)
    D = len(pts[0])
    prob = np.zeros((N, K))
    for k in range(K):
        xs = pts - np.tile(miu[k], (N, 1))
        isig = np.linalg.inv(sigma[k])
        t = np.sum(xs.dot(isig) * xs, 1)
        coef = np.power(2 * math.pi, -D / 2) * np.sqrt(np.linalg.det(isig))
        prob[:, k] = coef * np.exp(-0.5 * t)
    return prob

def gmm(pts, K):
    N = len(pts)
    D = len(pts[0])
    miu, pi, sigma = initialize_params(pts, K)
    Lprev = -999999
    it = 1
    while True:
        prob = estimate(pts, K, miu, pi, sigma) # n by k
        gamma = prob * np.tile(pi, (N, 1))
        gamma = gamma / np.tile(np.sum(gamma, 1), (K, 1)).transpose() # n by k
        Nk = np.sum(gamma, 0) # 1 by k

        miu = np.diag(1 / Nk).dot(gamma.transpose()).dot(pts) # k by d
        pi = Nk / N

        for k in range(K):
            xs = pts - np.tile(miu[k], (N, 1))
            sigma[k] = (xs.transpose().dot(np.diag(gamma[:, k]).dot(xs))) / Nk[k]

        L = np.sum(np.log(prob.dot(pi.transpose())))
        print('Iteration %d: %f' % (it, L - Lprev))
        it += 1
        if L - Lprev < 1e-15: break
        Lprev = L
    return np.argmax(prob, axis=1)

class Pt():
    def __init__(self, id, d, N):
        self.d = np.array(d)
        self.prob = [0 for i in range(N)]
        self.id = id
        self.children = []
    
    def addProb(self, round):
        self.prob[round] += 1

    def getCluster(self):
        index, value = max(enumerate(self.prob), key=operator.itemgetter(1))
        return index, value

    def getDistance(self, pt):
        return np.linalg.norm(self.d - pt.d)
    
    def shiftSearch(self, id, pts, bandWidth=2):
        shift = np.array([0., 0.])
        nNeighbor = 0
        for pt in pts:
            dist = self.getDistance(pt)
            if dist < bandWidth:
                pt.prob[id] += 1
                shift += (pt.d - self.d)
                nNeighbor += 1
        return shift / nNeighbor
    
    def update(self, shift):
        self.d = self.d + shift


def generate_pts(data):
    pts = []
    for id, d in enumerate(data):
        pts.append(Pt(id, d, len(data)))
    return pts

def mergeDetect(pt, centers, r=3):
    for index, c in enumerate(centers):
        if c.getDistance(pt) < r:
            return True, index
    return False, -1    

def mean_shift(pts, bandWidth=2, eps=1e-6):
    centers = []
    
    for initialPt in pts:
        pt = copy.deepcopy(initialPt)
        while True:
            shift = pt.shiftSearch(pt.id, pts)
            pt.update(shift)
            if np.linalg.norm(shift) < eps:
                flag, index = mergeDetect(pt, centers)
                if flag:
                    centers[index].children.append(pt.id)
                else:
                    centers.append(pt)
                break
    mapping = {}
    for ci, c in enumerate(centers):
        mapping[c.id] = ci
        for childId in c.children:
            mapping[childId] = ci
    cluster = []
    for pt in pts:
        cluster.append(mapping[pt.getCluster()[0]])
    return cluster


def main():
    cluster_markers = ['rx', 'gx', 'bx', 'kx', 'kx', 'kx', 'kx', 'kx']
    data = generate_data()
    pts = generate_pts(data)
    cluster = mean_shift(pts)


    # cr = gmm(np.dstack((x, y))[0], len(Clusters))
    for i, d in enumerate(data):
        plt.plot(d[0], d[1], cluster_markers[cluster[i]])
    #plt.plot(x, y, 'rx')
    plt.show()

if __name__ == '__main__':
    main()