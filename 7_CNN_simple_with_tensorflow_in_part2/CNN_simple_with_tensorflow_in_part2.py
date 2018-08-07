# -*- coding: utf-8 -*-
"""
Created on Fri Jul 20 09:22:14 2018

@author: a
"""
#%%================================Part 1：Convolutional Neural Networks: Step by Step
#package
import numpy as np
import h5py
import matplotlib.pyplot as plt

%matplotlib inline
plt.rcParams['figure.figsize'] = (5.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

%load_ext autoreload
%autoreload 2

np.random.seed(1)
#%%------------------------------2 - Outline of the Assignment

"""
-Convolution functions, including: 
 Zero Padding
 Convolve window
 Convolution forward
 Convolution backward (optional)
-Pooling functions, including: 
 Pooling forward
 Create mask
 Distribute value
 Pooling backward (optional)
"""

#%%-------------------------------3 - Convolutional Neural Networks
"""
input->>CONV1->>relu->>POOL1->>CONV2->>relu->>POOL2->>FC->>softmax
"""
#3.1 Zero-Padding
def zero_pad(X,pad):
    """
    X -shape(m,n_H,n_W,n_C),a batch of m images
    pad
    X_pad - shape(m,n_H+2*pad,n_W+2*pad,n_C) 
    """
    X_pad = np.pad(X,((0,0),(pad,pad),(pad,pad),(0,0)),'constant')
    return X_pad
"""
np.random.seed(1)
x = np.random.randn(4, 3, 3, 2)
x_pad = zero_pad(x, 2)
print ("x.shape =", x.shape)
print ("x_pad.shape =", x_pad.shape)
print ("x[1,1] =", x[1,1])
print ("x_pad[1,1] =", x_pad[1,1])

fig, axarr = plt.subplots(1, 2)
axarr[0].set_title('x')
axarr[0].imshow(x[0,:,:,0])
axarr[1].set_title('x_pad')
axarr[1].imshow(x_pad[0,:,:,0])
"""
#3.2 single step of convolution
def conv_single_step(a_slice_prev,W,b):
    """
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    W -- Weight parameters contained in a window - matrix of shape (f, f, n_C_prev)
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Z -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    """
    s = a_slice_prev * W
    Z = np.sum(s)
    Z = Z + b
    return Z
"""
np.random.seed(1)
a_slice_prev = np.random.randn(4, 4, 3)
W = np.random.randn(4, 4, 3)
b = np.random.randn(1, 1, 1)

Z = conv_single_step(a_slice_prev, W, b)
print("Z =", Z)
"""

#3.3 - Convolutional Neural Networks - Forward pass
def conv_forward(A_prev,W,b,hparameters):
    """
    Implements the forward propagation for a convolution function

    Arguments:
    A_prev -- output activations of the previous layer, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f, f, n_C_prev, n_C)
    b -- Biases, numpy array of shape (1, 1, 1, n_C)
    hparameters -- python dictionary containing "stride" and "pad"

    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward() function
    """    
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (f,f,n_C_prev,n_C) = W.shape
    
    stride = hparameters["stride"]
    pad = hparameters["pad"]

    n_H = int((n_H_prev - f + 2*pad)/stride + 1)
    n_W = int((n_W_prev - f + 2*pad)/stride + 1)
    
    Z = np.zeros((m,n_H,n_W,n_C))
    
    A_prev_pad = zero_pad(A_prev,pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    #find the location of the current convolution on
                    vert_start = stride*h
                    vert_end = vert_start + f
                    horiz_start = stride*w
                    horiz_end = horiz_start + f
                    
                    a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    
                    Z[i,h,w,c] = conv_single_step(a_slice_prev,W[:,:,:,c],b[:,:,:,c])
                    
    assert(Z.shape == (m,n_H,n_W,n_C))
    
    cache = (A_prev,W,b,hparameters)
    
    return Z,cache
    #need to be activated:A = activation(Z）
"""
np.random.seed(1)
A_prev = np.random.randn(10,4,4,3)
W = np.random.randn(2,2,3,8)
b = np.random.randn(1,1,1,8)
hparameters = {"pad" : 2,
               "stride": 2}

Z, cache_conv = conv_forward(A_prev, W, b, hparameters)
print("Z's mean =", np.mean(Z))
print("Z[3,2,1] =", Z[3,2,1])
print("cache_conv[0][1][2][3] =", cache_conv[0][1][2][3])
"""
        
#4.Pooling Layer
def pool_forward(A_prev,hparameters,mode="max"):
    """
    Implements the forward pass of the pooling layer

    Arguments:
    A_prev -- Input data, numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    hparameters -- python dictionary containing "f" and "stride"
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    A -- output of the pool layer, a numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache used in the backward pass of the pooling layer, contains the input and hparameters 
    """        
        
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape

    f = hparameters["f"]
    stride = hparameters["stride"]

    n_H = int((n_H_prev-f)/stride + 1)
    n_W = int((n_W_prev-f)/stride + 1)
    n_C = n_C_prev

    A = np.zeros((m,n_H,n_W,n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    #find the location of the POOL
                    vert_start = stride*h
                    vert_end = vert_start + f
                    horiz_start = stride*w
                    horiz_end = horiz_start + f
                    
                    a_prev_slice = A_prev[i,vert_start:vert_end,horiz_start:horiz_end,c]
    
                    if mode== "max":
                        A[i,h,w,c] = np.max(a_prev_slice)
                    elif mode == "average":
                        A[i,h,w,c] = np.mean(a_prev_slice)
                        
    assert(A.shape==(m,n_H,n_W,n_C))
    
    cache = (A_prev,hparameters)
    
    return A,cache

"""
np.random.seed(1)
A_prev = np.random.randn(2, 4, 4, 3)
hparameters = {"stride" : 2, "f": 3}

A, cache = pool_forward(A_prev, hparameters)
print("mode = max")
print("A =", A)
print()
A, cache = pool_forward(A_prev, hparameters, mode = "average")
print("mode = average")
print("A =", A)
"""                
                        
#5.1 - Backpropagation in convolutional neural networks (OPTIONAL / UNGRADED)

def conv_backward(dZ,cache):
    """
    Implement the backward propagation for a convolution function

    Arguments:
    dZ -- gradient of the cost with respect to the output of the conv layer (Z), numpy array of shape (m, n_H, n_W, n_C)
    cache -- cache of values needed for the conv_backward(), output of conv_forward()

    Returns:
    dA_prev -- gradient of the cost with respect to the input of the conv layer (A_prev),
               numpy array of shape (m, n_H_prev, n_W_prev, n_C_prev)
    dW -- gradient of the cost with respect to the weights of the conv layer (W)
          numpy array of shape (f, f, n_C_prev, n_C)
    db -- gradient of the cost with respect to the biases of the conv layer (b)
          numpy array of shape (1, 1, 1, n_C)
    """
    (A_prev,W,b,hparameters) = cache
    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (f,f,n_C_prev,n_C) = W.shape
    stride = hparameters["stride"]
    pad = hparameters["pad"]
    (m,n_H,n_W,n_C) = dZ.shape
    
    dA_prev = np.zeros((m,n_H_prev,n_W_prev,n_C_prev))
    dW = np.zeros((f,f,n_C_prev,n_C))
    db = np.zeros((1,1,1,n_C))
    
    A_prev_pad = zero_pad(A_prev,pad)
    dA_prev_pad = zero_pad(dA_prev,pad)
    
    for i in range(m):
        a_prev_pad = A_prev_pad[i,:,:,:]
        da_prev_pad = dA_prev_pad[i,:,:,:]

        for h in range(n_H):
            for w in range(n_W):
               for c in range(n_C):
                   vert_start = stride * h
                   vert_end = vert_start + f
                   horiz_start = stride * w
                   horiz_end = horiz_start + f
                   
                   a_slice = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                    
                   da_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:] += W[:,:,:,c] * dZ[i,h,w,c]
                   dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                   db[:,:,:,c] += dZ[i, h, w, c]
        dA_prev[i,:,:,:] = da_prev_pad[pad:-pad,pad:-pad,:]

    assert(dA_prev.shape==(m,n_H_prev,n_W_prev,n_C_prev))
    
    return dA_prev,dW,db

"""
np.random.seed(1)
dA, dW, db = conv_backward(Z, cache_conv)
print("dA_mean =", np.mean(dA))
print("dW_mean =", np.mean(dW))
print("db_mean =", np.mean(db))
"""

#5.2 Pooling layer - backward pass
def create_mask_from_window(x):
    """
    Creates a mask from an input matrix x, to identify the max entry of x.

    Arguments:
    x -- Array of shape (f, f)

    Returns:
    mask -- Array of the same shape as window, contains a True at the position corresponding to the max entry of x.
    """
    mask = (x==np.max(x))
    
    return mask
    
"""
np.random.seed(1)
x = np.random.randn(2,3)
mask = create_mask_from_window(x)
print('x = ', x)
print("mask = ", mask)
"""

def distribute_value(dz,shape):
    """
    Distributes the input value in the matrix of dimension shape

    Arguments:
    dz -- input scalar
    shape -- the shape (n_H, n_W) of the output matrix for which we want to distribute the value of dz

    Returns:
    a -- Array of size (n_H, n_W) for which we distributed the value of dz
    """
    (n_H, n_W) = shape
    average = dz/(n_H*n_W)
    a = average*np.ones(shape)

    return a
    
"""
a = distribute_value(2, (2,2))
print('distributed value =', a)
"""

def pool_backward(dA,cache,mode="max"):
    """
    Implements the backward pass of the pooling layer

    Arguments:
    dA -- gradient of cost with respect to the output of the pooling layer, same shape as A
    cache -- cache output from the forward pass of the pooling layer, contains the layer's input and hparameters 
    mode -- the pooling mode you would like to use, defined as a string ("max" or "average")

    Returns:
    dA_prev -- gradient of cost with respect to the input of the pooling layer, same shape as A_prev
    """
    (A_prev,hparameters) = cache
    stride = hparameters["stride"]
    f = hparameters["f"]

    (m,n_H_prev,n_W_prev,n_C_prev) = A_prev.shape
    (m,n_H,n_W,n_C) = dA.shape
    
    dA_prev = np.zeros(np.shape(A_prev))
    
    for i in range(m):
        a_prev = A_prev[i,:,:,:]
    
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    vert_start = stride * h
                    vert_end = vert_start + f
                    horiz_start = stride * w
                    horiz_end = horiz_start + f
                    
                    if mode == "max":
                        a_prev_slice = a_prev[vert_start:vert_end,horiz_start:horiz_end,c]
                        mask = create_mask_from_window(a_prev_slice)
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += np.multiply(mask,dA[i,h,w,c])
                        
                    elif mode == "average":
                        da = dA[i,h,w,c]
                        shape = (f,f)
                        dA_prev[i,vert_start:vert_end,horiz_start:horiz_end,c] += distribute_value(da,shape)
    assert(dA_prev.shape == A_prev.shape)
    
    return dA_prev
    
"""
np.random.seed(1)
A_prev = np.random.randn(5, 5, 3, 2)
hparameters = {"stride" : 1, "f": 2}
A, cache = pool_forward(A_prev, hparameters)
dA = np.random.randn(5, 4, 2, 2)

dA_prev = pool_backward(dA, cache, mode = "max")
print("mode = max")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1])  
print()
dA_prev = pool_backward(dA, cache, mode = "average")
print("mode = average")
print('mean of dA = ', np.mean(dA))
print('dA_prev[1,1] = ', dA_prev[1,1]) 
"""

#%%=========================Part 2：Convolutional Neural Networks: Application
#with tensorflow
#1.0- TensorFlow model
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
import tensorflow as tf
from tensorflow.python.framework import ops
from cnn_utils import *

%matplotlib inline
np.random.seed(1)

# Loading the data (signs)
X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
index = 6
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))

X_train = X_train_orig/255.
X_test = X_test_orig/255.
Y_train = convert_to_one_hot(Y_train_orig, 6).T
Y_test = convert_to_one_hot(Y_test_orig, 6).T
print ("number of training examples = " + str(X_train.shape[0]))
print ("number of test examples = " + str(X_test.shape[0]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))
conv_layers = {}

"""
number of training examples = 1080
number of test examples = 120
X_train shape: (1080, 64, 64, 3)
Y_train shape: (1080, 6)
X_test shape: (120, 64, 64, 3)
Y_test shape: (120, 6)
"""

#%%--------------------1.1 - Create placeholders

def create_placeholders(n_H0,n_W0,n_C0,n_y):
    """
    Creates the placeholders for the tensorflow session.

    Arguments:
    n_H0 -- scalar, height of an input image
    n_W0 -- scalar, width of an input image
    n_C0 -- scalar, number of channels of the input
    n_y -- scalar, number of classes

    Returns:
    X -- placeholder for the data input, of shape [None, n_H0, n_W0, n_C0] and dtype "float"
    Y -- placeholder for the input labels, of shape [None, n_y] and dtype "float"
    """
    X = tf.placeholder(tf.float32,shape=[None,n_H0,n_W0,n_C0])
    Y = tf.placeholder(tf.float32,shape=[None,n_y])
    
    return X,Y
    
"""
X, Y = create_placeholders(64, 64, 3, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))
"""

#%%---------------1.2 - Initialize parameters

def initialize_parameters():
    """
    Initializes weight parameters to build a neural network with tensorflow. The shapes are:
                        W1 : [4, 4, 3, 8]
                        W2 : [2, 2, 8, 16]
    Returns:
    parameters -- a dictionary of tensors containing W1, W2
    """
    
    tf.set_random_seed(1)
    
    W1 = tf.get_variable("W1",[4,4,3,8],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    W2 = tf.get_variable("W2",[2,2,8,16],initializer=tf.contrib.layers.xavier_initializer(seed=0))
    
    parameters = {"W1": W1,
                  "W2": W2}
                  
    return parameters
    
"""
tf.reset_default_graph()
with tf.Session() as sess_test:
    parameters = initialize_parameters()
    init = tf.global_variables_initializer()
    sess_test.run(init)
    print("W1 = "+str(parameters["W1"].eval()[1,1,1]))
    print("W2 = "+str(parameters["W2"].eval()[1,1,1]))
"""    
    


#%%------------------------1.2 - Forward propagation
"""
CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED
"""
def forward_propagation(X,parameters):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "W2"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    #CONV2D:stride 1 ,padding "same"
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    A1 = tf.nn.relu(Z1)
    #MAXPOOL:window 8*8,stride 8,padding 'same'
    P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')
    
    #CONV2D:stride 1,padding 'same'
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
    A2 = tf.nn.relu(Z2)
    #MAXPOOL:window 4*4, stride 4, padding 'same'
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')
    
    #flatten
    P2 = tf.contrib.layers.flatten(P2)
    
    #FULLY-CONNECTED without non_linear activation:6 neurons in output layers
    Z3 = tf.contrib.layers.fully_connected(P2,6,activation_fn=None)
    
    return Z3
"""
tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    X,Y = create_placeholders(64,64,3,6)#X:(?,64,64,3)   Y:(?,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(Z3,feed_dict={X:np.random.randn(2,64,64,3),Y:np.random.randn(2,6)})
    print("Z3 = "+str(a))
"""
#%%---------------------1.3 - Compute cost

def compute_cost(Z3,Y):
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = Z3, labels = Y))
    return cost
"""
tf.reset_default_graph()
with tf.Session() as sess:
    np.random.seed(1)
    X,Y = create_placeholders(64,64,3,6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X,parameters)
    cost = compute_cost(Z3,Y)
    init = tf.global_variables_initializer()
    sess.run(init)
    a = sess.run(cost,{X:np.random.randn(4,64,64,3),Y:np.random.randn(4,6)})
    print("Cost = "+str(a))
"""
#%%----------------------------1.4 Model

def model(X_train,Y_train,X_test,Y_test,learning_rate=0.009,
          num_epochs = 100, minibatch_size = 64, print_cost = True, to_predict=False):
    """
    Implements a three-layer ConvNet in Tensorflow:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> FULLYCONNECTED

    Arguments:
    X_train -- training set, of shape (None, 64, 64, 3)
    Y_train -- test set, of shape (None, n_y = 6)
    X_test -- training set, of shape (None, 64, 64, 3)
    Y_test -- test set, of shape (None, n_y = 6)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    train_accuracy -- real number, accuracy on the train set (X_train)
    test_accuracy -- real number, testing accuracy on the test set (X_test)
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (m,n_H0,n_W0,n_C0) = X_train.shape
    n_y = Y_train.shape[1]
    costs = []
    
    X,Y = create_placeholders(n_H0,n_W0,n_C0,n_y)
    
    parameters = initialize_parameters()
    
    Z3 = forward_propagation(X,parameters)
    
    cost = compute_cost(Z3,Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            minibatch_cost = 0.
            num_minibatches = int(m / minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)
            
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
                _ , temp_cost = sess.run([optimizer,cost],feed_dict={X:minibatch_X,Y:minibatch_Y})
                
                minibatch_cost += temp_cost / num_minibatches
                
            if print_cost == True and epoch%5 == 0:
                print("Cost after epoch %i:%f"%(epoch,minibatch_cost))
            if print_cost == True and epoch%1 == 0:
                costs.append(minibatch_cost)
                
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iteration(per ten)')
        plt.title("Learning rate ="+str(learning_rate))
        plt.show()
        
        parameters = sess.run(parameters)#########################################important!!!!
        
        predict_op = tf.argmax(Z3,1)
        correct_prediction = tf.equal(predict_op,tf.argmax(Y,1))
        
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        print(accuracy)
        train_accuracy = accuracy.eval({X:X_train,Y:Y_train})
        test_accuracy = accuracy.eval({X:X_test,Y:Y_test})
        print("Train Accuracy:",train_accuracy)
        print("Test Accuracy:",test_accuracy)
        
        return train_accuracy,test_accuracy,parameters


_, _, parameters = model(X_train, Y_train, X_test, Y_test,num_epochs = 100)



#%%test own 
fname = "images/thumbs_up.jpg"
image = np.array(ndimage.imread(fname, flatten=False))
image = scipy.misc.imresize(image, size=(64,64))
plt.imshow(image)
my_image = image.reshape((1,64,64,3))

my_image_prediction = predict(my_image/255, parameters)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))

#%%------predict function
#
def forward_propagation_for_predict(X,parameters):
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    Z1 = tf.nn.conv2d(X,W1,strides=[1,1,1,1],padding='SAME')
    A1 = tf.nn.relu(Z1)
    P1 = tf.nn.max_pool(A1,ksize=[1,8,8,1],strides=[1,8,8,1],padding='SAME')    
    Z2 = tf.nn.conv2d(P1,W2,strides=[1,1,1,1],padding='SAME')
    A2 = tf.nn.relu(Z2)
    P2 = tf.nn.max_pool(A2,ksize=[1,4,4,1],strides=[1,4,4,1],padding='SAME')    
    P2 = tf.contrib.layers.flatten(P2)
    Z3 = tf.contrib.layers.fully_connected(P2,6,activation_fn=None)
    
    return Z3

def predict(X,parameters):
    W1 = tf.convert_to_tensor(parameters["W1"])
    W2 = tf.convert_to_tensor(parameters["W2"])
    
    (_,n_H0,n_W0,n_C0) = X.shape
    
    params = {"W1":W1,"W2":W2}

    x = tf.placeholder("float",[None,n_H0,n_W0,n_C0])

    z3 = forward_propagation_for_predict(x,params)
    p = tf.argmax(z3,1)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        pred = sess.run(p,feed_dict={x:X})

    return pred 


"""我大哥卓贤的代码
def test_new_images(parameters, new_images):
    X = tf.placeholder(tf.float32, shape=[None, new_images.shape[1], new_images.shape[2],
                                          new_images.shape[3]])

    Z3 = forward_propagation(X, parameters)
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        predict_out = tf.argmax(Z3, 1)
        out = sess.run(predict_out,feed_dict={X:new_images})
    return out
"""

                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                        
                    
                    
                    
                    
                    
        
        
        
        
        
        
        
        
        










