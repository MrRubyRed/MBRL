# -*- coding: utf-8 -*-
"""
Created on Thu May 26 14:13:20 2016

@author: cusgadmin
"""

import tensorflow as tf

def BuildRNN(lsizes = None, state_dim = None, action_dim = None, H_depth = 1):
    
    input_s = tf.placeholder(tf.float32,shape=(None,state_dim));
    u_l = tf.placeholder(tf.float32,shape=(None,action_dim*H_depth));
    y = tf.placeholder(tf.float32,shape=(None,state_dim*H_depth));
    y_l = [tf.slice(y,[0,i*state_dim],[-1,state_dim]) for i in range(H_depth)];   
  
    lw = [];
    lb = [];
    reg = 0.0;
    for i in range(len(lsizes) - 1):
        lw.append(0.1*tf.Variable(tf.random_uniform([lsizes[i],lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="H"+str(i)));
        lb.append(0.1*tf.Variable(tf.random_uniform([1,lsizes[i + 1]],-1.0,1.0,dtype=tf.float32),name="B"+str(i)));
        reg = reg + tf.reduce_sum(tf.abs(lw[-1])) + tf.reduce_sum(tf.abs(lb[-1]));

    #First Part of the NN                                     
    curr_u = tf.slice(u_l,[0,0],[-1,action_dim]);
    next_i = tf.concat([input_s,curr_u],1);                               
    l = [];                                     
    l.append(tf.nn.sigmoid(tf.add(tf.matmul(next_i,lw[0]), lb[0]),name="A"+str(0)))
    for i in range(len(lw)-2):
        l.append(tf.nn.sigmoid(tf.add(tf.matmul(l[-1],lw[i+1]), lb[i+1]),name="A"+str(i)));

    l.append(tf.add(tf.matmul(l[-1],lw[-1]), lb[-1],name="A_end"));
    
    # Creating full "RNN"            
    Big_l = [];              
    Big_l.append(l[-1]);
            
    for i in range(H_depth-1):
        
        curr_u = tf.slice(u_l,[0,(i+1)*action_dim],[-1,action_dim]);
        next_i = tf.concat([Big_l[-1],curr_u],1); 
        
        l.append(tf.nn.sigmoid(tf.add(tf.matmul(next_i,lw[0]), lb[0])))
        for i in range(len(lw)-2):
            l.append(tf.nn.sigmoid(tf.add(tf.matmul(l[-1],lw[i+1]), lb[i+1])));
    
        l.append(tf.add(tf.matmul(l[-1],lw[-1]), lb[-1]));
        Big_l.append(l[-1]);                          
                              
                              
                              
    return input_s,u_l,y,y_l,Big_l,lw, lb, reg


        
        
    
    