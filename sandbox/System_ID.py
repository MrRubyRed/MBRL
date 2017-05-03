"""
Demo of system identification on an inverted pendulum, using one step dynamics.

@author: David Fridovich-Keil
@email: dfk@eecs.berkeley.edu
"""

import gym
import tensorflow as tf
from mbrl.dynamics.dynamics import Dynamics
import numpy as np
import pickle

# Make a gym environment.
env = gym.make("Pendulum-v0")
STATE_DIM = 3
ACTION_DIM = 1

# Create a Dynamics neural net.
dyn = Dynamics(STATE_DIM, ACTION_DIM, [STATE_DIM + ACTION_DIM, 20, 10, STATE_DIM])

# Create loss functor (RMS error) and set up optimizer.
rms = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(
    tf.sub(dyn.output_, dyn.next_state_)), 1, keep_dims=True)))
learning_rate = 0.01
momentum = 0.995
train_step = tf.train.MomentumOptimizer(
    learning_rate=learning_rate, momentum=momentum).minimize(rms)

# Set up tf session.
sess = tf.Session()
init = tf.initialize_all_variables()
sess.run(init)

# Create a bunch of rollouts
NUM_ROLLOUTS = 300
TIME_HORIZON = 20

sa = np.zeros((NUM_ROLLOUTS * TIME_HORIZON, STATE_DIM + ACTION_DIM))
targets = np.zeros((NUM_ROLLOUTS * TIME_HORIZON, STATE_DIM))
for ii in range(NUM_ROLLOUTS):
    s = env.reset()

    for jj in range(TIME_HORIZON):
        a = np.random.uniform(-2.0, 2.0, (1,))
        sa[ii*TIME_HORIZON + jj, :STATE_DIM] = s
        sa[ii*TIME_HORIZON + jj, STATE_DIM:] = a

        step = env.step(a)
        targets[ii * TIME_HORIZON, :] = step[0]
        s = step[0]

# Training.
BATCH_SIZE = 20
NUM_TRAIN_STEPS = 10000
for ii in range(NUM_TRAIN_STEPS):
    indices = np.random.randint(NUM_ROLLOUTS * TIME_HORIZON, size=BATCH_SIZE)

    sess.run(train_step, feed_dict={dyn.sa_ : sa[indices],
                                    dyn.next_state_ : targets[indices]})

    if np.mod(ii, 50) == 0:
        print("RMS error at iteration %d is %f" %
                  (ii, sess.run(rms, feed_dict={dyn.sa_ : sa,
                                                dyn.next_state_ : targets})))
