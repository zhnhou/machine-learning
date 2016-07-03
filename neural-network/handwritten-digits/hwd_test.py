from mnist import *
from hwd_neuralnet import *

path = './data/'

fetch_mnist_train(path)
image, label = load_mnist_train(path)

## image: n_example * n_feature
## label: n_example

num_output = 10
num_feature = image.shape[1]

hwd_nn = NeuralNet(num_output, num_feature, num_hidden=30, lambda1=0.0, lambda2=0.1, decrease_rate=1e-5)
hwd_nn.nn_learn(image, label)
