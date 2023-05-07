import numpy as np
from activation import *
class Layer(object):
    def __init__(self,neurons,pre_neurons,activation,weights,add_bias= True):
        self.a = activation
        self.neurons = neurons
        self.pre_neurons = pre_neurons
        self.weights = weights if weights is not None else np.random.randn(self.neurons, self.pre_neurons+1)
        self.add_bias = add_bias

    def forwardPropogate(self):
        
        
        if self.add_bias == True:
            bias = np.array([1])
            X = np.append(bias,self.a)
            X = np.matrix(X).T
            
        z = np.dot(self.weights,X)
        output = activation.sigmoid(z)
        pair = (X, self.weights)
        
        return output,pair
    
    def backPropogate(self,delta,pair):
        a, weights = pair
       
        A = np.dot(weights.T,delta)
        
        
        one_matrix = np.ones((a.shape))

        B = np.multiply(a,(one_matrix-a))
        
        delta = np.multiply(A,B)
        delta = np.delete(delta,0)
        
        return np.matrix(delta).T
    

