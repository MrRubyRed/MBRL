# -*- coding: utf-8 -*-

import gym
import tensorflow as tf
from FAuxFuncsRNN import BuildRNN
import numpy as np
import pickle

env = gym.make("Pendulum-v0");

# ======== Tensorflow init
H_depth = 5;
state_dim = 3;
action_dim = 1;
input_s,u_l,y,y_l,Big_l,lw,lb,reg = BuildRNN(lsizes = [3+1,10,3], state_dim = state_dim, action_dim = action_dim, H_depth = H_depth);

L_l = [tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y_l[i],Big_l[i])),1,keep_dims=True))) for i in range(H_depth)]
L = tf.reduce_sum(L_l);
#L = tf.sqrt(tf.reduce_mean(tf.reduce_sum(tf.square(tf.sub(y,NN)),1,keep_dims=True)));

nu = 0.01;
#train_step = tf.train.GradientDescentOptimizer(nu).minimize(L);
mom = 0.95;
train_step = tf.train.MomentumOptimizer(learning_rate=nu,momentum=mom).minimize(L)
#train_step = tf.train.RMSPropOptimizer(learning_rate=nu,momentum=mom).minimize(L)

sess = tf.Session();
init = tf.initialize_all_variables();
sess.run(init);
# ========
num_r = 300;
T_hor = 20;

list_states = [];
list_actions = [];
for i in range(num_r):
    s = env.reset();
    l_s = [s];
    l_a = [];
    for j in range(T_hor):
        a = np.random.uniform(-2.0,2.0,(1,));
        s = env.step(a);
        l_s.append(s[0]);
        l_a.append(a);
    list_states.append(l_s);
    list_actions.append(l_a);

#    input_s = tf.placeholder(tf.float32,shape=(None,state_dim));
#    u_l = tf.placeholder(tf.float32,shape=(None,action_dim*H_depth));
#    y = tf.placeholder(tf.float32,shape=(None,state_dim*H_depth));


output_NN = np.zeros((num_r*(T_hor-H_depth+1),state_dim*H_depth));
input_NNs = np.zeros((num_r*(T_hor-H_depth+1),state_dim));
input_NNa = np.zeros((num_r*(T_hor-H_depth+1),action_dim*H_depth));
for i in range(num_r):
    for j in range(T_hor-H_depth+1):
        output_NN[i*(T_hor-H_depth+1) + j,:] = np.concatenate(list_states[i][j+1:j+1+H_depth]);
        input_NNs[i*(T_hor-H_depth+1) + j,:] = list_states[i][j];
        input_NNa[i*(T_hor-H_depth+1) + j,:] = np.concatenate(list_actions[i][j:j+H_depth]);



# ===== Training
bts = 20;
for i in range(30000):
    tmp = np.random.randint(num_r*(T_hor-H_depth+1), size=bts);
    sess.run(train_step, feed_dict={input_s:input_NNs[tmp],y:output_NN[tmp],u_l:input_NNa[tmp]});
    if(np.mod(i,50) == 0):
        RMS = sess.run(L,{input_s:input_NNs,y:output_NN,u_l:input_NNa});
        print(str(RMS));



# ===== Testing
num_r = 300;
T_hor = 20;

list_states_ = [];
list_actions_ = [];
for i in range(num_r):
    s = env.reset();
    l_s = [s];
    l_a = [];
    for j in range(T_hor):
        a = np.random.uniform(-2.0,2.0,(1,));
        s = env.step(a);
        l_s.append(s[0]);
        l_a.append(a);
    list_states_.append(l_s);
    list_actions_.append(l_a);

#    input_s = tf.placeholder(tf.float32,shape=(None,state_dim));
#    u_l = tf.placeholder(tf.float32,shape=(None,action_dim*H_depth));
#    y = tf.placeholder(tf.float32,shape=(None,state_dim*H_depth));


output_NN_ = np.zeros((num_r*(T_hor-H_depth+1),state_dim*H_depth));
input_NNs_ = np.zeros((num_r*(T_hor-H_depth+1),state_dim));
input_NNa_ = np.zeros((num_r*(T_hor-H_depth+1),action_dim*H_depth));
for i in range(num_r):
    for j in range(T_hor-H_depth+1):
        output_NN_[i*(T_hor-H_depth+1) + j,:] = np.concatenate(list_states_[i][j+1:j+1+H_depth]);
        input_NNs_[i*(T_hor-H_depth+1) + j,:] = list_states_[i][j];
        input_NNa_[i*(T_hor-H_depth+1) + j,:] = np.concatenate(list_actions_[i][j:j+H_depth]);


RMS = sess.run([L_l[i] for i in range(H_depth)],{input_s:input_NNs_,y:output_NN_,u_l:input_NNa_});
print("Final Testing Error: " + str(RMS));

# Diverging trajectories?
Test_hor = 50;
Test_num = 1000;

in_nn = np.zeros((1,state_dim));
norm_errs = np.zeros((Test_num, Test_hor));
for n in range(Test_num):
    s = env.reset();

    if n % 100 == 1:
        print("Processing test run " + str(n) + "...");

    for j in range(Test_hor):
        a = np.random.uniform(-2.0,2.0,(1,));
        in_nn[0,:] = s;
        s_j_nn = sess.run(Big_l[0],{input_s:in_nn,u_l:np.concatenate((a[None],np.zeros((1,H_depth-1))),axis=1)});# np.array([[a[0]]])});
        s_j_env = env.step(a);

        norm_errs[n, j] = np.linalg.norm(s_j_nn - s_j_env[0]);
        #print("L_2 difference: " + str(np.linalg.norm(s_j_nn - s_j_env[0])));
        s = s_j_nn;

pickle.dump(norm_errs, open("norm_errs_H" + str(H_depth) + ".pkl", "wb"));
print("Saved to disk.");
