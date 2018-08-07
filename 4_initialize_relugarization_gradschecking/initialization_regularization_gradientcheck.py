# -*- coding: utf-8 -*-
"""
Created on Mon Jul 16 19:12:56 2018

@author: a
"""

#%%=======================================================Part 1： 初始化
"""
选择合适的初始值，可以：
-加速梯度下降的收敛
-增加梯度下降收敛到较低的训练和泛化误差的可能性
"""

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
#from init_utils import sigmoid, relu, compute_loss,forward_propagation, backward_propagation
from init_utils import *
#from init_utils import update_parameters, predict, load_dataset, plot_decision_boundary, predict_dec


%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = load_dataset()

#%%================================Neural Network Model
"""
You will use a 3-layer neural network (already implemented for you). Here are the initialization methods you will experiment with: 
- Zeros initialization – setting initialization = "zeros" in the input argument. 
- Random initialization – setting initialization = "random" in the input argument. This initializes the weights to large random values. 
- He initialization – setting initialization = "he" in the input argument. This initializes the weights to random values scaled according to a paper by He et al., 2015.
"""
def initialize_parameters_zeros(layers_dims):
    parameters = {}
    L = len(layers_dims)
    
    for l in range(1,L):
        parameters["W"+str(l)] = np.zeros((layers_dims[l],layers_dims[l-1]))
        parameters["b"+str(l)] = np.zeros((layers_dims[l],1))
        
    return parameters

def initialize_parameters_random(layers_dims):
    parameters = {}
    L = len(layers_dims)
    np.random.seed(3)
    
    for l in range(1,L):
        parameters["W"+str(l)] = np.random.randn(layers_dims[l],layers_dims[l-1])*10
        parameters["b"+str(l)] = np.zeros((layers_dims[l],1))
    
    return parameters
    
def initialize_parameters_he(layers_dims):
    parameters = {}
    L = len(layers_dims) - 1
    np.random.seed(3)
    
    for l in range(1,L+1):
        parameters["W"+str(l)] = np.sqrt(2.0/layers_dims[l-1])*np.random.randn(layers_dims[l],layers_dims[l-1])
        parameters["b"+str(l)] = np.zeros((layers_dims[l],1))
    return parameters

def model(X,Y,learning_rate = 0.01,num_iterations = 15000, print_cost = True, initialization = "he"):
    """
    initialization -- flag to choose which initialization to use ("zeros","random" or "he")
    
    return parameters learned by model
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],10,5,1]

    if initialization=="zeros":
        parameters = initialize_parameters_zeros(layers_dims)
    elif initialization=="random":
        parameters = initialize_parameters_random(layers_dims)
    elif initialization=="he":
        parameters = initialize_parameters_he(layers_dims)
        
    for i in range(0,num_iterations):
        a3,cache = forward_propagation(X,parameters)
        cost = compute_loss(a3,Y)
        grads = backward_propagation(X,Y,cache)
        parameters = update_parameters(parameters,grads,learning_rate)
        
        if print_cost and i%1000 == 0:
            print("Cost after iteration {}:{}".format(i,cost))
            costs.append(cost)
            
    # plot the loss
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (per hundreds)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()
    
    return parameters
    
#%%--------------1.zeros initialization
parameters = initialize_parameters_zeros([3,2,1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

#train
parameters = model(train_X,train_Y,initialization = "zeros")
print("On the train set:")
predictions_train = predict(train_X,train_Y,parameters)
print("On the test set:")
predictions_test = predict(test_X,test_Y,parameters)
"""
The performance is really bad, and the cost does not really decrease, 
and the algorithm performs no better than random guessing.
 Why? Lets look at the details of the predictions and the decision boundary:全部预测到0
"""
print ("predictions_train = " + str(predictions_train))
print ("predictions_test = " + str(predictions_test))

plt.title("Model with Zeros initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
#要注意：W的初值应该破坏对称性

#%%-----------------2.random initialization
parameters = initialize_parameters_random([3, 2, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

#train
parameters = model(train_X, train_Y, initialization = "random")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

"""
"""
print (predictions_train)
print (predictions_test)

plt.title("Model with large random initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
"""
In summary: 
- Initializing weights to very large random values does not work well. 
- Hopefully intializing with small random values does better. 
The important question is: how small should be these random values be? 
Lets find out in the next part!
"""
#%%-------------------------3.He initialization
parameters = initialize_parameters_he([2, 4, 1])
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

parameters = model(train_X, train_Y, initialization = "he")
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with He initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

"""
What you should remember from this notebook: 
- Different initializations lead to different results 
- Random initialization is used to break symmetry and make sure different hidden units can learn different things 
- Don’t intialize to values that are too large 
- He initialization works well for networks with ReLU activations.
"""

#%%===================================================Part 2：Regularization  正则化  泛化能力
"""
如果数据集没有很大，
同时在训练集上又拟合得很好，
但是在测试集的效果却不是很好，
这时候就要使用正则化来使得其拟合能力不会那么强。
"""
# import packages
import numpy as np
import matplotlib.pyplot as plt
#from reg_utils import sigmoid, relu, plot_decision_boundary, initialize_parameters, load_2D_dataset, predict_dec
#from reg_utils import compute_cost, predict, forward_propagation, backward_propagation, update_parameters
from reg_utils import *
import sklearn
import sklearn.datasets
import scipy.io
from testCases import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

train_X, train_Y, test_X, test_Y = load_2D_dataset()
"""
- If the dot is blue, it means the French player managed to hit the ball with his/her head 
- If the dot is red, it means the other team’s player hit the ball with their head
"""
#%%----------1. Non-regularized model
def model(X,Y,learning_rate = 0.3 , num_iterations = 30000, print_cost=True,lambd = 0, keep_prob = 1):
    """
    lambd -- regularization hyperparameter, scalar
    keep_prob - probability of keeping a neuron active during drop-out, scalar.
    return parameters
    """
    grads = {}
    costs = []
    m = X.shape[1]
    layers_dims = [X.shape[0],20,3,1]
    
    parameters = initialize_parameters(layers_dims)
    
    for i in range(0,num_iterations):#{
        if keep_prob == 1:
            a3,cache = forward_propagation(X,parameters)
        elif keep_prob <1:
            a3,cache = forward_propagation_with_dropout(X,parameters,keep_prob)
            
        if lambd == 0:
            cost = compute_cost(a3,Y)
        else:
            cost = compute_cost_with_regularization(a3,Y,parameters,lambd)
            
        assert(lambd==0 or keep_prob ==1)
        
        if lambd == 0 and keep_prob == 1:
            grads = backward_propagation(X,Y,cache)
        elif lambd != 0:
            grads = backward_propagation_with_regularization(X,Y,cache,lambd)
        elif keep_prob != 1:
            grads = backward_propagation_with_dropout(X,Y,cache,keep_prob)
            
        parameters = update_parameters(parameters,grads,learning_rate)
        
        if print_cost and i%10000==0:
            print("Cost after iteration {}:{}".format(i,cost))
        if print_cost and i%1000 ==0:
            costs.append(cost)#}
            
    #plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('iterations (x1,000)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    return parameters

#without any regularization
parameters = model(train_X, train_Y)
print ("On the training set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model without regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

"""
It is overfitting.
"""

#%%--------------2.L2 Regularization
def compute_cost_with_regularization(A3,Y,parameters,lambd):
    m = Y.shape[1]
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    W3 = parameters["W3"]

    cross_entropy_cost = compute_cost(A3,Y)
    
    L2_regularization_cost = (1./m*lambd/2)*(np.sum(np.square(W1)) + np.sum(np.square(W2)) + np.sum(np.square(W3)))
    
    cost = cross_entropy_cost + L2_regularization_cost
    
    return cost
"""
A3, Y_assess, parameters = compute_cost_with_regularization_test_case()
print("cost = " + str(compute_cost_with_regularization(A3, Y_assess, parameters, lambd = 0.1)))#cost = 1.78648594516
"""

def backward_propagation_with_regularization(X,Y,cache,lambd):
    m = X.shape[1]
    (Z1,A1,W1,b1,Z2,A2,W2,b2,Z3,A3,W3,b3) = cache
    dZ3 = A3 - Y
    
    dW3 = 1./m * np.dot(dZ3,A2.T) + lambd/m * W3
    db3 = 1./m * np.sum(dZ3,axis=1,keepdims = True)
    
    dA2 = np.dot(W3.T,dZ3)
    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = 1./m * np.dot(dZ2,A1.T) + lambd/m * W2
    db2 = 1./m * np.sum(dZ2,axis=1,keepdims = True)
    
    dA1 = np.dot(W2.T,dZ2)
    dZ1 = np.multiply(dA1,np.int64(A1>0))
    dW1 = 1./m * np.dot(dZ1,X.T) + lambd/m * W1
    db1 = 1./m * np.sum(dZ1,axis=1,keepdims = True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
                
    return gradients
"""
X_assess, Y_assess, cache = backward_propagation_with_regularization_test_case()

grads = backward_propagation_with_regularization(X_assess, Y_assess, cache, lambd = 0.7)
print ("dW1 = "+ str(grads["dW1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("dW3 = "+ str(grads["dW3"]))
"""

#run the model
parameters = model(train_X, train_Y, lambd = 0.7)
print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with L2-regularization")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

"""
- L2 regularization makes your decision boundary smoother.
  If λλ is too large, it is also possible to “oversmooth”, 
  resulting in a model with high bias!!!
- The cost computation: 
  A regularization term is added to the cost 
- The backpropagation function: 
  There are extra terms in the gradients with respect to weight matrices 
- Weights end up smaller (“weight decay”): 
- Weights are pushed to smaller values.
"""

#%%-----------------3 - Dropout
###反向随机失活
def forward_propagation_with_dropout(X,parameters,keep_prob = 0.5):
    np.random.seed(1)
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    W3 = parameters["W3"]
    b3 = parameters["b3"]

    Z1 = np.dot(W1,X)+b1
    A1 = relu(Z1)
    D1 = np.random.rand(A1.shape[0],A1.shape[1])
    D1 = D1 < keep_prob
    A1 = A1 * D1
    A1 = A1 / keep_prob
    
    Z2 = np.dot(W2,A1)+b2
    A2 = relu(Z2)
    D2 = np.random.rand(A2.shape[0],A2.shape[1])
    D2 = D2 < keep_prob
    A2 = A2 * D2
    A2 = A2 / keep_prob
    
    Z3 = np.dot(W3,A2) + b3
    A3 = sigmoid(Z3)
    
    cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

    return A3,cache
    
"""
X_assess, parameters = forward_propagation_with_dropout_test_case()

A3, cache = forward_propagation_with_dropout(X_assess, parameters, keep_prob = 0.7)
print ("A3 = " + str(A3))
"""

def backward_propagation_with_dropout(X,Y,cache,keep_prob):
    m = X.shape[1]
    (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache
    
    dZ3 = A3 - Y
    dW3 = 1./m * np.dot(dZ3, A2.T)
    db3 = 1./m * np.sum(dZ3, axis=1, keepdims=True)
    
    dA2 = np.dot(W3.T,dZ3)
    dA2 = dA2 * D2 / keep_prob
    dZ2 = np.multiply(dA2,np.int64(A2>0))
    dW2 = 1./m * np.dot(dZ2, A1.T)
    db2 = 1./m * np.sum(dZ2, axis=1, keepdims=True)
    
    dA1 = np.dot(W2.T,dZ2)
    dA1 = dA1 * D1 / keep_prob
    dZ1 = np.multiply(dA1,np.int64(A1>0))
    dW1 = 1./m * np.dot(dZ1, X.T)
    db1 = 1./m * np.sum(dZ1, axis=1, keepdims=True)
    
    gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3,"dA2": dA2,
                 "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1, 
                 "dZ1": dZ1, "dW1": dW1, "db1": db1}
                 
    return gradients
    
"""
X_assess, Y_assess, cache = backward_propagation_with_dropout_test_case()

gradients = backward_propagation_with_dropout(X_assess, Y_assess, cache, keep_prob = 0.8)

print ("dA1 = " + str(gradients["dA1"]))
print ("dA2 = " + str(gradients["dA2"]))
"""

#run the model
parameters = model(train_X, train_Y, keep_prob = 0.86, learning_rate = 0.3)

print ("On the train set:")
predictions_train = predict(train_X, train_Y, parameters)
print ("On the test set:")
predictions_test = predict(test_X, test_Y, parameters)

plt.title("Model with dropout")
axes = plt.gca()
axes.set_xlim([-0.75,0.40])
axes.set_ylim([-0.75,0.65])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)

"""
- A common mistake when using dropout is to use it both in training and testing. You should use dropout (randomly eliminate nodes) only in training. 
- Deep learning frameworks like tensorflow, PaddlePaddle, keras or caffe come with a dropout layer implementation. Don’t stress - you will soon learn some of these frameworks.
- Dropout is a regularization technique. 
- You only use dropout during training. Don’t use dropout (randomly eliminate nodes) during test time. 
- Apply dropout both during forward and backward propagation. 
- During training time, divide each dropout layer by keep_prob to keep the same expected value for the activations. 
  For example, if keep_prob is 0.5, then we will on average shut down half the nodes, 
  so the output will be scaled by 0.5 since only the remaining half are contributing to the solution. 
  Dividing by 0.5 is equivalent to multiplying by 2.
  Hence, the output now has the same expected value.
  You can check that this works even when keep_prob is other values than 0.5.
"""

#conclusions
"""
What we want you to remember from this notebook: 
- Regularization will help you reduce overfitting. 
- Regularization will drive your weights to lower values. 
- L2 regularization and Dropout are two very effective regularization techniques.
"""

#%%========================================Part 3：Gradient Checking
import numpy as np
from testCases import *
from gc_utils import dictionary_to_vector, vector_to_dictionary, gradients_to_vector
from reg_utils import sigmoid, relu

#略过，详情看以下链接
#https://blog.csdn.net/koala_tree/article/details/78137306




















