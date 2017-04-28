import numpy as np

def gini_score(x, y):

    def gini(s):
        _, freq = np.unique(s, return_counts=True)
        prob = freq/s.size
        prob *= prob
        return prob.sum()
    
    score, split = 0, 0
    for splitter in np.unique(x):
        score_ = gini(y[x <= splitter]) + gini(y[x > splitter])
        if score_ > score:
            score, split = score_, splitter

    return (score - gini(y), split)

def chi_score(x, y):

    score, split = 0, 0
    p_T = np.count_nonzero(y == 1)/y.size
    p_F = 1 - p_T
    for splitter in np.unique(x):
        score_ = 0
        # left class
        y_l = y[x <= splitter]
        if y_l.size:
            pl_T = np.count_nonzero(y_l == 1)/y_l.size
            pl_F = 1 - pl_T
            score_ += ((p_T - pl_T)**2/p_T)**0.5 if pl_T > 0 else 0
            score_ += ((p_F - pl_F)**2/p_F)**0.5 if pl_F > 0 else 0
        # right class
        y_r = y[x > splitter]
        if y_r.size:
            pr_T = np.count_nonzero(y_r == 1)/y_r.size
            pr_F = 1 - pr_T
            score_ += ((p_T - pr_T)**2/p_T)**0.5 if pl_T > 0 else 0
            score_ += ((p_F - pr_F)**2/p_F)**0.5 if pl_F > 0 else 0
        if score_ > score:
            score, split = score_, splitter

    return (score, split)

def entropy_score(x, y):

    def entropy(s):
        _, freq = np.unique(s, return_counts=True)
        prob = freq/s.size
        return -sum([p*np.log(p) for p in prob])

    score, split = 0, 0

    for splitter in np.unique(x):
        score_ = entropy(y) - (np.count_nonzero(x <= splitter) * entropy(y[x <= splitter]) + np.count_nonzero(x > splitter) * entropy(y[x > splitter]))/x.size
        if score_ > score:
            score, split = score_, splitter
        
    return (score, split)

class DecisionTree:
    def __init__(self, score_func = gini_score, max_depth = 20):
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
        scores = np.array([self.score(dim_data, y) for dim_data in x]).T
        split_dim = np.argmax(scores[0, :])
        split_val = scores[1, split_dim]

        if scores[0, split_dim] < (10 ** -6):
            print('Useless')
            print(scores, x.shape, y.shape)
            return y
        
        # recursively build the tree
        l =  self.build_tree(x[:, x[split_dim, :] <= split_val], y[x[split_dim, :] <= split_val], depth + 1)
        r =  self.build_tree(x[:, x[split_dim, :] > split_val], y[x[split_dim, :] > split_val], depth + 1)
        children = (l, r)
        
        return (split_dim, split_val, children)
        
    def predict(self, x):
        
        assert self.root is not None  # check if training was done
        
        d, n = x.shape
        prediction = np.empty((n, ))
        
        for i in range(n):
            datap = x[:, i]
            node = self.root
            while type(node) == tuple:
                split_dim = node[0]
                split_val = node[1]
                l, r = node[2]
                if datap[split_dim] <= split_val:
                    node = l
                else:
                    node = r
            prediction[i] = np.random.choice(node) if len(node) else np.random.choice([-1, 1])
        
        return prediction
