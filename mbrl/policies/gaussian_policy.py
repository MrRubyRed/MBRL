"""
A neural network that models a Gaussian policy. Input is state, output is the
mean of a Gaussian distribution over actions (variance is held constant).

@author: David Fridovich-Keil
@email: dfk@eecs.berkeley.edu
"""

import tensorflow as tf

class GaussianPolicy:

    # Constructor. TODO!
    def __init__(self, state_dim = None, action_dim = None, layer_sizes = None,
                 variance = 1.0):
        pass
