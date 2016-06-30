import numpy as np
import os
import struct

def fetch_mnist_train(path):
    
    image_file = 'train-images-idx3-ubyte.gz'
    label_file = 'train-labels-idx1-ubyte.gz'

    web_path = 'http://yann.lecun.com/exdb/mnist/'

    if ( not os.path.isfile(path+image_file) ):
        os.system("wget -O "+path+"/"+image_file+" "+web_path+image_file)
        os.system("gzip "+path+"/"+image_file+" -d") # decompress and delete .gz file
    if ( not os.path.isfile(path+label_file) ):
        os.system("wget -O "+path+"/"+label_file+" "+web_path+label_file)
        os.system("gzip "+path+"/"+label_file+" -d")

## load the mnist training data
def load_mnist_train(path):
    image_file = path+"/train-images-idx3-ubyte"
    label_file = path+"/train-labels-idx1-ubyte"

    with open(label_file) as lb_unit:
        magic, n = struct.unpack('>II', lb_unit.read(8))
        labels = np.fromfile(lb_unit, dtype=np.uint8)

    with open(image_file) as im_unit:
        magic, num, rows, cols = struct.unpack('>IIII', im_unit.read(16))
        images = np.fromfile(im_unit, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

## load the mnist test data
def load_mnist_test(path):
    image_file = path+"/t10k-images-idx3-ubyte"
    label_file = path+"/t10k-labels-idx1-ubyte"

    with open(label_file) as lb_unit:
        magic, n = struct.unpack('>II', lb_unit.read(8))
        labels = np.fromfile(lb_unit, dtype=np.uint8)

    with open(image_file) as im_unit:
        magic, num, rows, cols = struct.unpack('>IIII', im_unit.read(16))
        images = np.fromfile(im_unit, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels

