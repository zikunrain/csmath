import math
import numpy as np
import matplotlib.pyplot as plt
import operator
import copy

def generateGaussian2D(N, meanX, meanY, var):
    x = np.random.normal(loc=meanX, scale=var, size=N)
    y = np.random.normal(loc=meanY, scale=var, size=N)
    return x, y


def generateData():
    x0, y0 = generateGaussian2D(300, 0, 0, 1.5)
    x1, y1 = generateGaussian2D(200, 7, 2, 1)
    x2, y2 = generateGaussian2D(200, 3, 5, 1.5)
    data = np.dstack((np.concatenate((x0, x1, x2)), np.concatenate((y0, y1, y2))))[0]
    return data


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
        print(initialPt.id, '/ %d' % len(pts))
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
    cluster_markers = ['bx', 'gx', 'rx', 'cx', 'mx', 'yx', 'kx', 'wx']
    data = generateData()
    pts = generate_pts(data)
    cluster = mean_shift(pts)
    for i, d in enumerate(data):
        plt.plot(d[0], d[1], cluster_markers[cluster[i]])
    plt.show()

if __name__ == '__main__':
    main()