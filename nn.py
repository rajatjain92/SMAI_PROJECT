import numpy as np

class NeuralNet:
    def __init__(self, inp_n, hidden_n, out_n):
        self.inp_n = inp_n
        self.hidden_n = hidden_n
        self.out_n = out_n
        
        # initialize randomly based on number of input units
        self.w_ji = (np.random.random((hidden_n, inp_n)) - 1)*(2/np.sqrt(inp_n))
        self.w_kj = (np.random.random((out_n, hidden_n)) - 1)*(2/np.sqrt(hidden_n))


    def sig(self, x):
        """Activation function. Used sigmoid by default""" 
        return 1/(1 + np.exp(-x))

    
    def sig_(self, x):
        """Derivative of activation function"""
        x = self.sig(x)
        return x*(1 - x)

    
    def train(self, data, t, thresh = 0.95):
        """Train the weights on multiple samples"""
        eeta = 0.1
        inp_n = self.inp_n
        hidden_n = self.hidden_n
        out_n = self.out_n
        n = data.shape[1]
    
        # verify dimensionality
        assert data.shape == (inp_n, n)
        assert t.shape == (out_n, n)
    
        loopnct = 0
        while loopnct < 1000:
            loopnct += 1
        
            # feed forward
            x = data
            net_j = self.w_ji.dot(x)
            y = self.sig(net_j)
            net_k = self.w_kj.dot(y)
            z = self.sig(net_k)
            
            # back propagation
            delta_k = (t - z) * self.sig_(net_k)
            dw_kj = delta_k.dot(y.T)
            delta_j = self.w_kj.T.dot(delta_k) * self.sig_(net_j)
            dw_ji = delta_j.dot(x.T)
            # print(loopnct)
            
            self.w_ji += eeta * dw_ji
            self.w_kj += eeta * dw_kj
        
            # DONOT overfit
            if self.verify(data, t) > thresh:
                break
        print('Loop count: ', loopnct)

    def predict(self, data):
        x = data
        net_j = self.w_ji.dot(x)
        y = self.sig(net_j)
        net_k = self.w_kj.dot(y)
        z = self.sig(net_k)
        return z
    
    def verify(self, data, t):
        z = self.predict(data) > 0.5
        t = t > 0.5
        n = t.shape[1]
        correct = 0
        for i in range(n):
            if (z[:, i] == t[:, i]).all():
                correct += 1
        return correct/n
