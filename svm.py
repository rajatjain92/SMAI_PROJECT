import numpy as np
import cvxopt

def linear_kernel(x, y):
    return np.dot(x, y)

def poly_kernel(x, y, p = 2):
    return (1 + np.dot(x, y))**p

def rbf_kernel(x, y, gamma = 1):
    return np.exp(-(gamma * np.linalg.norm(x - y)))

class MySVM:
    def __init__(self, kernel = np.dot, c = 100):
        self.kernel = kernel
        self.c = c
        self.alpha = None

    def gram(self, x):
        d, n = x.shape
        toret = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                toret[i, j] = self.kernel(x[:, i], x[:, j])
        return toret

    def find_sv(self, x, y):
        d, n = x.shape

        k = self.gram(x)

        # Minimize 1/2*x.T*P*x + q.T*x
        P = np.outer(y, y) * k
        q = -np.ones((n, 1))

        # G*x <= h
        G = np.vstack((-np.identity(n), np.identity(n)))
        h = np.vstack((np.zeros((n, 1)), self.c * np.ones((n, 1))))

        # A*x = b
        A = np.reshape(y, (1, n))
        b = np.zeros(1)

        solution = cvxopt.solvers.qp(*map(lambda x: cvxopt.matrix(x, tc = 'd'), [P, q, G, h, A, b]))

        # Lagrange multipliers
        return np.array(solution['x']).flatten()

    def train(self, x, y):
        alpha = self.find_sv(x, y)
        sv_ind = alpha > alpha.max()/100

        # store support vector info
        self.alpha = alpha[sv_ind]
        self.n = self.alpha.size
        self.data = x[:, sv_ind]
        self.label = y[sv_ind]
        k = self.gram(self.data)
        self.b = self.label.mean() + sum([(self.alpha * self.label * k[:, i]).sum() for i in range(self.n)])

        return sv_ind

    def predict(self, x, raw = False):
        d, n = x.shape
        assert self.alpha is not None
        prediction = np.zeros((n, )) + self.b
        for i in range(n):
            for j in range(self.n):
                prediction[i] += self.alpha[j] * self.label[j] * self.kernel(x[:, i], self.data[:, j])

        return prediction if raw else np.sign(prediction)
