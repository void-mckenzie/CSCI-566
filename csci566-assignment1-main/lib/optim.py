from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


""" Super Class """
class Optimizer(object):
    """
    This is a template for implementing the classes of optimizers
    """
    def __init__(self, net, lr=1e-4):
        self.net = net  # the model
        self.lr = lr    # learning rate

    """ Make a step and update all parameters """
    def step(self):
        #### FOR RNN / LSTM ####
        if hasattr(self.net, "preprocess") and self.net.preprocess is not None:
            self.update(self.net.preprocess)
        if hasattr(self.net, "rnn") and self.net.rnn is not None:
            self.update(self.net.rnn)
        if hasattr(self.net, "postprocess") and self.net.postprocess is not None:
            self.update(self.net.postprocess)
        
        #### MLP ####
        if not hasattr(self.net, "preprocess") and \
           not hasattr(self.net, "rnn") and \
           not hasattr(self.net, "postprocess"):
            for layer in self.net.layers:
                self.update(layer)


""" Classes """
class SGD(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4):
        self.net = net
        self.lr = lr

    def update(self, layer):
        for n, dv in layer.grads.items():
            layer.params[n] -= self.lr * dv


class SGDM(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-4, momentum=0.0):
        self.net = net
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}  # last update of the velocity

    def update(self, layer):
        #############################################################################
        # TODO: Implement the SGD + Momentum                                        #
        #############################################################################
        for n, dv in layer.grads.items():
            prev_vt = self.velocity.get(n, 0)
            v_t = self.momentum * prev_vt - self.lr * dv
            layer.params[n] += v_t
            self.velocity[n] = v_t

        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class RMSProp(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-2, decay=0.99, eps=1e-8):
        self.net = net
        self.lr = lr
        self.decay = decay
        self.eps = eps
        self.cache = {}  # decaying average of past squared gradients

    def update(self, layer):
        #############################################################################
        # TODO: Implement the RMSProp                                               #
        #############################################################################
        for n, dv in layer.grads.items():
            prev_sq_grad = self.cache.get(n, 0)
            grad_sq = self.decay * prev_sq_grad + (1 - self.decay) * np.square(dv)
            layer.params[n] -= (self.lr * dv ) / np.sqrt(grad_sq + self.eps)
            self.cache[n] = grad_sq
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################


class Adam(Optimizer):
    """ Some comments """
    def __init__(self, net, lr=1e-3, beta1=0.9, beta2=0.999, t=0, eps=1e-8):
        self.net = net
        self.lr = lr
        self.beta1, self.beta2 = beta1, beta2
        self.eps = eps
        self.mt = {}
        self.vt = {}
        self.t = t

    def update(self, layer):
        #############################################################################
        # TODO: Implement the Adam                                                  #
        #############################################################################
        self.t += 1
        for layer in self.net.layers:
            for n, dv in layer.grads.items():
                if n in self.mt:
                    self.mt[n] = self.beta1 * self.mt[n] + (1-self.beta1) * dv
                else:
                    self.mt[n] = (1-self.beta1) * dv
                if n in self.vt:
                    self.vt[n] = self.beta2 * self.vt[n] + (1-self.beta2) * dv**2
                else:
                    self.vt[n] = (1-self.beta2) * dv**2
                layer.params[n] -= (self.lr * self.mt[n] / (1-self.beta1**(self.t))) \
                    / (np.sqrt(self.vt[n]/(1-self.beta2**(self.t))) + self.eps)
        pass
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################