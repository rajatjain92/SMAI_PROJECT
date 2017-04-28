import numpy as np
import decisionTree as dt
import svm
import nn

class RandomForest:
    def __init__(self, ratio = 10, hidden_num = 5):
        self.rat = ratio
        self.trees = [dt.DecisionTree(score_func=dt.gini_score, max_depth = 5)
                      ,dt.DecisionTree(score_func=dt.gini_score, max_depth = 10)
                      ,dt.DecisionTree(score_func=dt.chi_score, max_depth = 5)
                      ,dt.DecisionTree(score_func=dt.chi_score, max_depth = 10)
                      ,dt.DecisionTree(score_func=dt.chi_score, max_depth = 20)
                      ,dt.DecisionTree(score_func=dt.entropy_score, max_depth = 5)
                      ,dt.DecisionTree(score_func=dt.entropy_score, max_depth = 10)
                      ,svm.MySVM(kernel = lambda x, y: svm.rbf_kernel(x, y, 0.01), c = 10)]
        self.num_trees = len(self.trees)
        self.tree_score = np.ones((self.num_trees,))
        self.nn = nn.NeuralNet(self.num_trees, hidden_num, 1)
    
    def train(self, x, y):
        d, n = x.shape
        for i in range(self.num_trees):
            rand_perm = np.random.permutation(n)
            tr_perm = rand_perm[:n//self.rat]
            tr_x = x[:, tr_perm]
            tr_y = y[tr_perm]
            self.trees[i].train(tr_x, tr_y)

        # train the nn to decide tree sizes
        tree_predict = np.vstack([tree.predict(x) for tree in self.trees])
        self.nn.train(tree_predict, y.reshape((1, n)))

        # assign weights based on score
        for i in range(self.num_trees):
            self.tree_score[i] = np.count_nonzero(tree_predict[i, :] == y)

    def predict(self, x, version = 'NN'):
        assert version in ('EQ', 'NN', 'WT')
        
        if version == 'EQ':  # output most voted prediction
            predictions = [tree.predict(x) for tree in self.trees]
            prediction = np.zeros((x.shape[1],))
            for pre in predictions:
                prediction += pre
            prediction = np.sign(prediction)
            prediction[prediction == 0] = np.random.choice([-1, 1], prediction.shape)[prediction == 0]
        elif version == 'NN':  # based on the neural network weights
            tree_predict = np.vstack([tree.predict(x) for tree in self.trees])
            prediction = (self.nn.predict(tree_predict) > 0.001) * 2 - 1
        elif version == 'WT':
            prediction = np.zeros((x.shape[1],))
            for i in range(self.num_trees):
                prediction += self.tree_score[i] * self.trees[i].predict(x)
            prediction = np.sign(prediction)
            prediction[prediction == 0] = np.random.choice([-1, 1], prediction.shape)[prediction == 0]
        return prediction
