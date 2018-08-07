# -*- coding: utf-8 -*-
"""
Created on Thu Jul 19 09:47:19 2018

@author: a
"""

#===========================TensorFlow Tutorial

#%%====================================1 - Exploring the Tensorflow Library
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.framework import ops
from tf_utils import load_dataset, random_mini_batches, convert_to_one_hot, predict

%matplotlib inline
np.random.seed(1)

#%%single example of using tensorflow
#--------------compute loss = (yhat - y)^2
#1
y_hat = tf.constant(36,name = 'y_hat')
y = tf.constant(39,name = 'y')
#2
loss = tf.Variable((y-y_hat)**2 , name = 'loss')
#3
init = tf.global_variables_initializer()
#4
with tf.Session() as session:
    session.run(init)
#5
    print(session.run(loss))
    
"""Writing and running programs in TensorFlow has the following steps:
1.Create Tensors (variables) that are not yet executed/evaluated.
2.Write operations between those Tensors.
3.Initialize your Tensors.
4.Create a Session.
5.Run the Session. This will run the operations you’d written above.
初始化变量，创建一个session，把operation放进session里面计算
"""

#------------定义一个变量：可以实时赋值，利用feed_dict
sess = tf.Session()
x = tf.placeholder(tf.int64,name='x')
print(sess.run(2 * x, feed_dict = {x:3}))
sess.close()
    
#%%------------------------------------------1.1 - Linear function    
"""
computing the following equation: Y = WX+b
Exercise: Compute WX+b where W,X and b are drawn from a random normal distribution. 
          W is of shape (4, 3), X is (3,1) and b is (4,1). 
- tf.matmul(…, …) to do a matrix multiplication 
- tf.add(…, …) to do an addition 
- np.random.randn(…) to initialize randomly    
"""

def linear_function():
    np.random.seed(1)
    
    X = tf.constant(np.random.randn(3,1),name = 'X')
    W = tf.constant(np.random.randn(4,3),name = 'W')
    b = tf.constant(np.random.randn(4,1),name = 'b')
    Y = tf.add(tf.matmul(W,X),b)
    
    session = tf.Session()
    result = session.run(Y)
    
    session.close()
    
    return result

print("result = "+str(linear_function()))
    
#%%------------------------------1.2 - Computing the sigmoid
def sigmoid(z):
    x = tf.placeholder(tf.float32,name = 'x')
    sigmoid = tf.sigmoid(x)
    
    with tf.Session() as sess:
        result = sess.run(sigmoid,feed_dict={x:z})
    
    return result
 
print ("sigmoid(0) = " + str(sigmoid(0)))
print ("sigmoid(12) = " + str(sigmoid(12)))
    
#%%------------------------------1.3 - Computing the Cost

def cost(logit,labels):
    z = tf.placeholder(tf.float32,name='z')
    y = tf.placeholder(tf.float32,name='y')
    
    cost = tf.nn.sigmoid_cross_entropy_with_logits(logits = z, labels = y)
    
    sess = tf.Session()
    cost = sess.run(cost,feed_dict={z:logit,y:labels})
    
    sess.close()
    
    return cost
    
logits = sigmoid(np.array([0.2,0.4,0.7,0.9]))
cost = cost(logits, np.array([0,0,1,1]))
print ("cost = " + str(cost))
    
#%%--------------------------1.4 - Using One Hot encodings    
"""
分类变量（哑变量）：C个类别的标签
"""
def one_hot_matrix(labels,C):
    """
    labels:0,1,2,3,...,C-1
    """
    C = tf.constant(value = C,name = 'C')
    
    one_hot_matrix = tf.one_hot(labels,C,axis=0)
    
    sess = tf.Session()
    one_hot = sess.run(one_hot_matrix)
    sess.close()
    
    return one_hot
    
labels = np.array([1,2,3,0,2,1])
one_hot = one_hot_matrix(labels, C = 4)
print ("one_hot = " + str(one_hot))

#%%-------------------------------1.5 - Initialize with zeros and ones

def ones(shape):
    ones = tf.ones(shape)
    
    sess = tf.Session()
    ones = sess.run(ones)
    sess.close()
    
    return ones

print ("ones = " + str(ones([3])))


#%%===============================2 - Building your first neural network in tensorflow
#Problem statement: SIGNS Dataset

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Example of a picture
index = 0
plt.imshow(X_train_orig[index])
print ("y = " + str(np.squeeze(Y_train_orig[:, index])))


# Flatten the training and test images
X_train_flatten = X_train_orig.reshape(X_train_orig.shape[0], -1).T
X_test_flatten = X_test_orig.reshape(X_test_orig.shape[0], -1).T
# Normalize image vectors
X_train = X_train_flatten/255.
X_test = X_test_flatten/255.
# Convert training and test labels to one hot matrices
Y_train = convert_to_one_hot(Y_train_orig, 6)
Y_test = convert_to_one_hot(Y_test_orig, 6)

print ("number of training examples = " + str(X_train.shape[1]))
print ("number of test examples = " + str(X_test.shape[1]))
print ("X_train shape: " + str(X_train.shape))
print ("Y_train shape: " + str(Y_train.shape))
print ("X_test shape: " + str(X_test.shape))
print ("Y_test shape: " + str(Y_test.shape))

"""
number of training examples = 1080
number of test examples = 120
X_train shape: (12288, 1080) -- 12288 := 64*64*3(RGB)
Y_train shape: (6, 1080)     -- 6:sign 0,1,2,3,4,5(6 in total)
X_test shape: (12288, 120)
Y_test shape: (6, 120)
"""

#%%Model
"""
LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SOFTMAX
The SIGMOID output layer has been converted to a SOFTMAX. 
A SOFTMAX layer generalizes SIGMOID to when there are more than two classes.
"""
#%%------------------------------------2.1 - Create placeholders
def create_placeholders(n_x,n_y):
    X = tf.placeholder(tf.float32,shape = [n_x, None])
    #use None because it let's us be flexible on the number of examples you will for the placeholders.
    #In fact, the number of examples during test/train is different.
    Y = tf.placeholder(tf.float32,shape = [n_y, None])
    
    return X,Y
"""
X, Y = create_placeholders(12288, 6)
print ("X = " + str(X))
print ("Y = " + str(Y))
"""
#%%----------------------------------2.2 - Initializing the parameters
"""
Implement the function below to initialize the parameters in tensorflow. 
You are going use Xavier Initialization for weights and Zero Initialization for biases. 
The shapes are given below. 
"""
def initialize_parameters():
    """
    the model layers_dims:12288,25,12,6,1
    """
    tf.set_random_seed(1)
    
    W1 = tf.get_variable("W1",[25,12288],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b1 = tf.get_variable("b1",[25,1],initializer = tf.zeros_initializer())
    W2 = tf.get_variable("W2",[12,25],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b2 = tf.get_variable("b2",[12,1],initializer = tf.zeros_initializer())
    W3 = tf.get_variable("W3",[6,12],initializer = tf.contrib.layers.xavier_initializer(seed = 1))
    b3 = tf.get_variable("b3",[6,1],initializer = tf.zeros_initializer())
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
                  
    return parameters
"""
tf.reset_default_graph()
with tf.Session() as sess:
    parameters = initialize_parameters()
    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))
"""
#%%-------------------------------2.3 - Forward propagation in tensorflow

def forward_propagation(X,parameters):
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = tf.add(tf.matmul(W1,X),b1)
    A1 = tf.nn.relu(Z1)
    Z2 = tf.add(tf.matmul(W2,A1),b2)
    A2 = tf.nn.relu(Z2)
    Z3 = tf.add(tf.matmul(W3,A2),b3)
    
    return Z3
"""
tf.reset_default_graph()
with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    print("Z3 = " + str(Z3))
"""
"""
注意到这里没有和以往一样保留缓冲池cache，这在后面BP的时候可以知道原因
"""
#%%-------------------------------------2.4 Compute cost
def compute_cost(Z3,Y):
    logits = tf.transpose(Z3)
    labels = tf.transpose(Y)
    
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels = labels))
    
    return cost
    
"""
tf.reset_default_graph()

with tf.Session() as sess:
    X, Y = create_placeholders(12288, 6)
    parameters = initialize_parameters()
    Z3 = forward_propagation(X, parameters)
    cost = compute_cost(Z3, Y)
    print("cost = " + str(cost))
"""

#%%-----------------------------2.5 - Backward propagation & parameter updates
"""
 All the backpropagation and the parameters update is taken care of in 1 line of code. 
 It is very easy to incorporate this line in the model.
"""
optimizer = tf.train.GradientDescentOptimizer(learning_rate = learning_rate).minimize(cost)
#To make the optimization you would do:
#_ , c = sess.run([optimizer, cost], feed_dict={X: minibatch_X, Y: minibatch_Y})

#%%-----------------------------2.6 - Building the model

def model(X_train,Y_train,X_test,Y_test,learning_rate = 0.0001,
          num_epochs = 1500, minibatch_size = 32, print_cost = True):
    ops.reset_default_graph()
    tf.set_random_seed(1)
    seed = 3
    (n_x,m) = X_train.shape
    n_y = Y_train.shape[0]
    costs = []

    X,Y = create_placeholders(n_x,n_y)
    
    parameters = initialize_parameters()
    
    Z3 = forward_propagation(X,parameters)
    
    cost = compute_cost(Z3,Y)
    
    optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
    
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
        sess.run(init)
        
        for epoch in range(num_epochs):
            epoch_cost = 0.
            num_minibatches = int(m/minibatch_size)
            seed = seed + 1
            minibatches = random_mini_batches(X_train,Y_train,minibatch_size,seed)
            
            for minibatch in minibatches:
                (minibatch_X,minibatch_Y) = minibatch
                _ , minibatch_cost = sess.run([optimizer,cost],feed_dict = {X:minibatch_X,Y:minibatch_Y})
                
                epoch_cost = epoch_cost + minibatch_cost / num_minibatches
                
            if print_cost == True and epoch%100 == 0:
                print("Cost after epoch %i: %f"%(epoch,epoch_cost))
            if print_cost == True and epoch%5 == 0:
                costs.append(epoch_cost)
        
        # plot the cost
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per tens)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
        
        #save the parameters in a variable
        parameters = sess.run(parameters)
        print("parameters have been trained")

        correct_prediction = tf.equal(tf.argmax(Z3),tf.argmax(Y))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction,"float"))
        
        print ("Train Accuracy:", accuracy.eval({X: X_train, Y: Y_train}))
        print ("Test Accuracy:", accuracy.eval({X: X_test, Y: Y_test}))

        return parameters
"""
Run the following cell to train your model! On our machine it takes about 5 minutes. 
Your “Cost after epoch 100” should be 1.016458. 
If it’s not, don’t waste time; 
interrupt the training by clicking on the square (⬛) in the upper bar of the notebook, 
and try to correct your code. 
If it is the correct cost, take a break and come back in 5 minutes!
"""
"""#run
parameters = model(X_train, Y_train, X_test, Y_test)
"""



'''
Insights: 
- Your model seems big enough to fit the training set well. 
  However, given the difference between train and test accuracy, 
  you could try to add L2 or dropout regularization to reduce overfitting. 
- Think about the session as a block of code to train the model. 
  Each time you run the session on a minibatch, it trains the parameters. 
  In total you have run the session a large number of times (1500 epochs) 
  until you obtained well trained parameters.
'''

#%%------------------2.7 - Test with your own image (optional / ungraded exercise)
"""
"""
import scipy
from PIL import Image
from scipy import ndimage

## START CODE HERE ## (PUT YOUR IMAGE NAME) 
my_image = "thumbs_up.jpg"
## END CODE HERE ##

# We preprocess your image to fit your algorithm.
fname = "images/" + my_image
image = np.array(ndimage.imread(fname, flatten=False))
my_image = scipy.misc.imresize(image, size=(64,64)).reshape((1, 64*64*3)).T
my_image_prediction = predict(my_image, parameters)

plt.imshow(image)
print("Your algorithm predicts: y = " + str(np.squeeze(my_image_prediction)))





