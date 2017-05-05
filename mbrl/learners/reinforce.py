"""
Implements the classic REINFORCE algorithm. Generates rollouts from the given
Dynamics and trains a GaussianPolicy.

@author: David Fridovich-Keil
@email: dfk@eecs.berkeley.edu
"""

from mbrl.dynamics.dynamics import Dynamics
from mbrl.policies.gaussian_policy import GaussianPolicy

import tensorflow as tf

class Reinforce:

    # Constructor. TODO!
    def __init__(self, dynamics, policy):
        pass
