# -*- coding: utf-8 -*-

import gym
import tensorflow as tf
from FAuxFuncs import BuildNN
import numpy as np
import pickle

env = gym.make("Pendulum-v0");

# ======== Tensorflow init
input_sa,y,NN,l,lw, lb, reg = BuildNN(lsizes = [3+1,20,10,3], state_dim = 3, action_dim = 1);

L = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y,NN)),1,keep_dims=True)));

nu = 0.01;
#train_step = tf.train.GradientDescentOptimizer(nu).minimize(L);
mom = 0.995;
train_step = tf.train.MomentumOptimizer(learning_rate=nu,momentum=mom).minimize(L)
#train_step = tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom).minimize(L)

sess = tf.Session();
init = tf.initialize_all_variables();
sess.run(init);
# ========

num_r = 300;
T_hor = 20;
inputs_NN = np.zeros((num_r*T_hor,3+1));
targs = np.zeros((num_r*T_hor,3));
for i in range(num_r):
    s = env.reset();
    for j in range(T_hor):
        a = np.random.uniform(-2.0,2.0,(1,));
        inputs_NN[i*T_hor + j,:-1] = s;
        inputs_NN[i*T_hor + j,-1] = a;
        s_n = env.step(a);
        targs[i*T_hor + j,:] = s_n[0];
        s = s_n[0];



# ===== Training
bts = 20;
for i in range(30000):
    tmp = np.random.randint(num_r*T_hor, size=bts);
    sess.run(train_step, feed_dict={input_sa:inputs_NN[tmp],y:targs[tmp]});
    if(np.mod(i,50) == 0):
        RMS = sess.run(L,{input_sa:inputs_NN,y:targs});
        print(str(RMS));



# ===== Testing
inputs_NN_ = np.zeros((num_r*T_hor,3+1));
targs_ = np.zeros((num_r*T_hor,3));
for i in range(num_r):
    s = env.reset();
    for j in range(T_hor):
        a = np.random.uniform(-2.0,2.0,(1,));
        inputs_NN_[i*T_hor + j,:-1] = s;
        inputs_NN_[i*T_hor + j,-1] = a;
        s_n = env.step(a);
        targs_[i*T_hor + j,:] = s_n[0];
        s = s_n[0];

RMS = sess.run(L,{input_sa:inputs_NN_,y:targs_});
print("Final Testing Error: " + str(RMS));

# Diverging trajectories?
Test_hor = 50;
Test_num = 1000;

in_nn = np.zeros((1,3+1));
norm_errs = np.zeros((Test_num, Test_hor));
for n in range(Test_num):
    s = env.reset();

    if n % 100 == 1:
        print("Processing test run " + str(n) + "...");

    for j in range(Test_hor):
        a = np.random.uniform(-2.0,2.0,(1,));
        in_nn[0,:-1] = s;
        in_nn[0,-1] = a;
        s_j_nn = sess.run(NN,{input_sa:in_nn})
        s_j_env = env.step(a);

        norm_errs[n, j] = np.linalg.norm(s_j_nn - s_j_env[0]);
#        print("L_2 difference: " + str(np.linalg.norm(s_j_nn - s_j_env[0])));
        s = s_j_nn;

pickle.dump(norm_errs, open("norm_errs_H1.pkl", "wb"));
print("Saved to disk.");
