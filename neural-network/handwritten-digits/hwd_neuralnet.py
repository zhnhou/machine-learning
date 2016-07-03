import numpy as np
from scipy.special import expit
import sys

class NeuralNet(object):

    ## input X: (n_example, n_feature)
    ## input y: (n_example)

    def __init__(self, num_output, num_feature, num_hidden=30, lambda1=0.0, lambda2=0.0, 
                 eta=0.001, alpha=0.0, epochs=1000, decrease_rate=0.00, shuffle=True, minibatches=50):
        self.num_output = num_output
        self.num_feature = num_feature
        self.num_hidden = num_hidden
        self.lambda1 = lambda1
        self.lambda2 = lambda2
        self.eta = eta
        self.alpha = alpha
        self.epochs = epochs
        self.decrease_rate = decrease_rate
        self.shuffle = shuffle
        self.minibatches = minibatches

        self.theta1, self.theta2 = self._initialize_weights()

    def _encode_labels(self, y, n_output):
        ones = np.zeros((n_output, y.shape[0]))

        for idx, val in enumerate(y):
            ones[val, idx] = 1.0
        return ones

    ## initialize the training parameters
    def _initialize_weights(self):
        theta1 = np.random.uniform(-1.0, 1.0, 
                 size=self.num_hidden*(self.num_feature+1)).reshape(self.num_hidden, self.num_feature+1)
        theta2 = np.random.uniform(-1.0, 1.0,
                 size=self.num_output*(self.num_hidden+1)).reshape(self.num_output, self.num_hidden+1)
        return theta1, theta2

    def _sigmoid(self, z):
        return expit(z)

    def _sigmoid_gradient(self, z):
        return expit(z) * (1 - expit(z))

    def _add_bias_unit(self, X):
        X_new = np.ones((X.shape[0], X.shape[1]+1))
        X_new[:,1:] = X

        return X_new

    def _L1_reg(self, lambda_, theta1, theta2):
        ## excluing the bias unit when doing regularization
        return lambda_/2.0 * (np.abs(theta1[:,1:]).sum() + np.abs(theta2[:,1:]).sum())

    def _L2_reg(self, lambda_, theta1, theta2):
        ## excluing the bias unit when doing regularization
        return lambda_/2.0 * (np.sum(theta1[:,1:]**2) + np.sum(theta2[:,1:]**2))

    def _feedforward(self, X, theta1, theta2):
        a1 = self._add_bias_unit(X) # n_example * (n_feature+1)
        z2 = a1.dot(theta1.T) # n_example * n_hidden ## here is different from the PML book
        a2 = self._sigmoid(z2) 
        a2 = self._add_bias_unit(a2) # n_example * (n_hidden+1)

        z3 = a2.dot(theta2.T) # n_example * n_output
        a3 = self._sigmoid(z3)
        
        return a1, z2, a2, z3, a3

    def _costfunction(self, y_enc, output, theta1, theta2):
        cost = np.sum(- y_enc * np.log(output) - (1-y_enc) * np.log(1 - output))

        ## regularization ##
        cost += self._L1_reg(self.lambda1, theta1, theta2) + self._L2_reg(self.lambda2, theta1, theta2)

        return cost


    ## implementing backpropagation
    def _gradient_backpropagation(self, a1, a2, a3, z2, y_enc, theta1, theta2):

        # y_enc - n_output * n_example
        delta3 = a3.T - y_enc

        z2 = self._add_bias_unit(z2) # n_example * (n_hidden+1)

        delta2 = theta2.T.dot(delta3) * self._sigmoid_gradient(z2.T) # (n_hidden+1) * n_example
        delta2 = delta2[1:,:]

        # a2: n_example * (n_hidden+1)
        # delta3: n_output * n_example
        grad2 = delta3.dot(a2) # n_output * n_hidden+1
        
        # delta2: n_hidden * n_example
        # a1: n_example * (n_feature+1)
        grad1 = delta2.dot(a1) # n_hidden * (n_feature+1)

        ## regularization
        grad1[:,1:] += theta1[:,1:] * (self.lambda1 + self.lambda2)
        grad2[:,1:] += theta2[:,1:] * (self.lambda1 + self.lambda2)

        return grad1, grad2

    def nn_learn(self, X, y):
        self.cost_ = []

        X_data = X.copy()
        y_data = y.copy()

        y_enc = self._encode_labels(y_data, self.num_output)

        delta1_prev = np.zeros(self.theta1.shape)
        delta2_prev = np.zeros(self.theta2.shape)

        for i in np.arange(self.epochs):
            ## adaptive learning rate
            self.eta *= 1.0/(1.0 + self.decrease_rate * i)

            sys.stderr.write('\rEpoch: %d/%d' % (i+1,self.epochs))
            sys.stderr.flush()

            if self.shuffle:
                idx = np.random.permutation(y_data.shape[0])
                
                X_data = X_data[idx] # note X_data is 2D array, this is very neat
                y_data = y_data[idx]

            mini = np.array_split(np.arange(y_data.shape[0]), self.minibatches)

            for idx in mini:
                
                # feedforward
                a1, z2, a2, z3, a3 = self._feedforward(X[idx], self.theta1, self.theta2)

                # a3 is already within mini samples
                cost = self._costfunction(y_enc[:,idx], a3.T, self.theta1, self.theta2)
                self.cost_.append(cost)

                # calculate grad via backpropagation
                grad1, grad2 = self._gradient_backpropagation(a1, a2, a3, z2, y_enc[:,idx], self.theta1, self.theta2)

                # update weights
                delta1, delta2 = self.eta * grad1, self.eta * grad2
                

                self.theta1 -= delta1 + self.alpha * delta1_prev
                self.theta2 -= delta2 + self.alpha * delta2_prev

                delta1_prev, delta2_prev = delta1, delta2
        
        return self

    def nn_predict(self, X):
        a1, z2, a2, z3, a3 = self._feedforward(X, self.theta1, self.theta2)

        # a3: n_example * n_output
        # axis=0 in the book but note the difference of how I get a2 in feedforward
        y_pred = np.argmax(a3, axis=1)

        return y_pred
