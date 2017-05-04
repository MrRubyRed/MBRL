"""
A neural network that models dynamics. Inputs are state and action, output is next state.

@author: David Fridovich-Keil
@email: dfk@eecs.berkeley.edu
"""

import tensorflow as tf

class Dynamics:

    # Constructor.
    def __init__(self, state_dim = None, action_dim = None, layer_sizes = None):
        self.state_ = tf.placeholder(tf.float32, shape=(None, state_dim), name="state")
        self.action_ = tf.placeholder(tf.float32, shape=(None, action_dim), name="action")
        self.next_state_ = tf.placeholder(tf.float32, shape=(None, state_dim), name="next_state")

        # Lists to store weights, biases, and layers. Also keep track of an L1 regularization term.
        self.weights_ = []
        self.biases_ = []
        self.layers_ = []
        self.reg_ = 0.0

        # Build up weights, biases, and regularization.
        for ii in range(len(layer_sizes) - 1):
            self.weights_.append(
                tf.Variable(tf.random_uniform([layer_sizes[ii], layer_sizes[ii + 1]],
                                              -0.1, 0.1, dtype=tf.float32),
                            name="H" + str(ii)))

            self.biases_.append(
                tf.Variable(tf.random_uniform([1, layer_sizes[ii + 1]],
                                               -0.1, 0.1, dtype=tf.float32),
                            name="B" + str(ii)))

            self.reg_ += (tf.reduce_sum(tf.abs(self.weights_[-1])) +
                          tf.reduce_sum(tf.abs(self.biases_[-1])))

        # Generate layers.
        self.layers_.append(
            tf.nn.sigmoid(tf.add(tf.add(tf.matmul(self.state_, self.weights_[0][:state_dim, :]),
                                        tf.matmul(self.action_, self.weights_[0][state_dim:, :])),
                                 self.biases_[0]),
                          name="A" + str(0)))

        for ii in range(len(self.weights_) - 2):
            self.layers_.append(
                tf.nn.sigmoid(tf.add(tf.matmul(self.layers_[-1], self.weights_[ii + 1]),
                                     self.biases_[ii + 1]),
                              name="A" + str(ii + 1)))

        self.layers_.append(
            tf.add(tf.matmul(self.layers_[-1], self.weights_[-1]), self.biases_[-1],
            name="A_end"))

        # Set output layer.
        self.output_ = self.layers_[-1]
