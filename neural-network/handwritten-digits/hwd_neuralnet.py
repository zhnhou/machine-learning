import numpy as np
from scipy.special import expit

class NeuralNet(object):

    def __init__(self, num_output, num_feature, num_hidden=30):
        self.num_output = num_output
        self.num_feature = num_feature
        self.num_hidden = num_hidden

    def _encode_labels(self, y, n_output):
        ones = np.zeros((n_output, y.shape[0]))

        for idx, val in enumerate(y):
            ones[val, idx] = 1.0
        return ones

    ## initialize the training parameters
    def _initialize_weights(self):
        theta1 = np.random.uniform(-1.0, 1.0, 
                 size=self.num_hidden*(self.num_feature+1)).reshape(self.num_hidden, self.num_feature+1)
        theta2 = np.random.uniform(-1.0, 1.0.
                 size=self.num_output*(self.num_hidden+1)).reshape(self.num_output, self.num_hidden+1)
        return theta1, theta2

    def _sigmoid(self, z):
        return expit(z)

    def _add_bias_unit(self, X):
        X_new = np.ones(X.shape[0], X.shape[1]+1)
        X_new[:,1:] = X

        return X_new

    def _feedforward(self, X, theta1, theta2):
        a1 = self._add_bias_unit(X)
        z2 = a1.dot(theta1.T)
        a2 = self._sigmoid(z2)
        a2 = self._add_bias_unit(a2)

        z3 = a2.dot(theta2.T)
        a3 = self._sigmoid(z3)
        
        return a1, z2, a2, z3, a3

    def _costfunction(self, y_enc, output, theta1, theta2):
        cost = - y_enc * np.log(output) - (1-y_enc) * np.log(1 - output)

