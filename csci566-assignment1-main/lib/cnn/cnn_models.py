from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from lib.mlp.fully_conn import *
from lib.mlp.layer_utils import *
from lib.cnn.layer_utils import *


""" Classes """
class TestCNN(Module):
    def __init__(self, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3,3,3,1,padding=0,name="conv1"),
            MaxPoolingLayer(pool_size=2, stride=2, name="maxpool1"),
            flatten(name="flat1"),
            #leaky_relu(name="lrelu1"),
            fc(input_dim=27,output_dim=5,init_scale=0.02,name="fc1")
            ########### END ###########
        )


class SmallConvolutionalNetwork(Module):
    def __init__(self, keep_prob=0, dtype=np.float32, seed=None):
        self.net = sequential(
            ########## TODO: ##########
            ConvLayer2D(3,5,16,1,padding=0,name="conv1"),
            MaxPoolingLayer(pool_size=2, stride=2, name="maxpool1"),
            ConvLayer2D(16,3,32,1,padding=0,name="conv2"),
            MaxPoolingLayer(pool_size=2, stride=2, name="maxpool2"),
            flatten(name="flat1"),
            leaky_relu(name="lrelu1"),
            dropout(0.75, seed=seed),
            fc(input_dim=1152,output_dim=10,init_scale=0.02,name="fc1")
            ########### END ###########
            ########### END ###########
        )
