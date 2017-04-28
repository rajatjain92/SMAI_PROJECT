import numpy as np
def entropy_score(x, y):

    def entropy(s):
        _, freq = np.unique(s, return_counts=True)
        prob = freq/s.size
        return -sum([p*np.log(p) for p in prob])

    score = entropy(y)

    val, freq = np.unique(x, return_counts=True)
    prob = freq/x.size

    for v, p in zip(val, prob):
        score -= p * entropy(y[x == v])

    return score

class DecisionTree:
    def __init__(self, score_func = entropy_score, max_depth = 2):
        self.score = score_func
        self.root = None
        self.max_depth = max_depth

    def train(self, x, y):
        self.root = self.build_tree(x, y)

    def no_split(self, y):
        return len(set(y)) in (0, 1)
        
    def build_tree(self, x, y, depth = 0):
        if self.no_split(y) or depth > self.max_depth:
            return y

        # select the best dimension and split
        scores = np.array([self.score(dim_data, y) for dim_data in x])
        split_dim = np.argmax(scores)

        if scores[split_dim] < (10 ** -6):
            return y
        
        # recursively build the tree
        node = {}
        for val in np.unique(x[split_dim, :]):
            node[val] = self.build_tree(x[:, x[split_dim, :] == val], y[x[split_dim, :] == val], depth + 1)
        
        return (split_dim, node)
        
    def predict(self, x):
        
        assert self.root is not None  # check if training was done

        d, n = x.shape
        prediction = np.empty((n, ))

        for i in range(n):
            datap = x[:, i]
            node = self.root
            while type(node) == tuple:
                split_dim = node[0]
                node = node[1][datap[split_dim]] if datap[split_dim] in node[1] else [0]

            prediction[i] = np.random.choice(node) if len(node) else 0 # randomly select from possibilities

        return prediction
