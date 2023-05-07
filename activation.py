import numpy as np
class activation:
    def sigmoid(z):
        # x: the original value of inputs
        return 1/(1 + np.exp(-z))