# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:13:20 2016

@author: cusgadmin
"""

import tensorflow as tf

def BuildNN(lsizes = None, state_dim = None, action_dim = None):
    
    input_sa = tf.placeholder(tf.float32,shape=(None,state_dim + action_dim),name="states");
    y = tf.placeholder(tf.float32,shape=(None,state_dim),name="y");   
    
    lw = [];
    lb = [];
    l = [];
    reg = 0.0;
    for i in range(len(lsizes) - 1):
        lw.append(0.1*tf.Variable(tf.random_uniform([lsizes[i],lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="H"+str(i)));
        lb.append(0.1*tf.Variable(tf.random_uniform([1,lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="B"+str(i)));
        reg = reg + tf.reduce_sum(tf.abs(lw[-1])) + tf.reduce_sum(tf.abs(lb[-1]));
        
    l.append(tf.nn.sigmoid(tf.add(tf.matmul(input_sa,lw[0]), lb[0]),name="A"+str(0)))
    for i in range(len(lw)-2):
        l.append(tf.nn.sigmoid(tf.add(tf.matmul(l[-1],lw[i+1]), lb[i+1]),name="A"+str(i)));

    l.append(tf.add(tf.matmul(l[-1],lw[-1]), lb[-1],name="A_end"));
    
    return input_sa,y,l[-1],l,lw, lb, reg


        
        
    
    