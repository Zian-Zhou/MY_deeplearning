# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 15:17:58 2018

@author: a
"""

"""
各种优化算法：
Mini-batch梯度下降、指数加权平均、Momentum梯度下降、RMSprop、Adam优化算法、衰减学习率、局部最优
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import math
import sklearn
import sklearn.datasets

from opt_utils import load_params_and_grads, initialize_parameters, forward_propagation, backward_propagation
from opt_utils import compute_cost, predict, predict_dec, plot_decision_boundary, load_dataset
from testCases import *

%matplotlib inline
plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

#%%===============================1 - Gradient Descent
def update_parameters_with_gd(parameters, grads, learning_rate):
    """
    Update parameters using one step of gradient descent

    Arguments:
    parameters -- python dictionary containing your parameters to be updated:
                    parameters['W' + str(l)] = Wl
                    parameters['b' + str(l)] = bl
    grads -- python dictionary containing your gradients to update each parameters:
                    grads['dW' + str(l)] = dWl
                    grads['db' + str(l)] = dbl
    learning_rate -- the learning rate, scalar.

    Returns:
    parameters -- python dictionary containing your updated parameters 
    """

    L = len(parameters) // 2 # number of layers in the neural networks

    # Update rule for each parameter
    for l in range(L):
        ### START CODE HERE ### (approx. 2 lines)
        parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - learning_rate*grads["dW" + str(l+1)]
        parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - learning_rate*grads["db" + str(l+1)]
        ### END CODE HERE ###

    return parameters
"""
parameters, grads, learning_rate = update_parameters_with_gd_test_case()

parameters = update_parameters_with_gd(parameters, grads, learning_rate)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
"""

#Batch Gradient Descent
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    # Forward propagation
    a, caches = forward_propagation(X, parameters)
    # Compute cost.
    cost = compute_cost(a, Y)
    # Backward propagation.
    grads = backward_propagation(a, caches, parameters)
    # Update parameters.
    parameters = update_parameters(parameters, grads)
    
#Stochastic Gradient Descent(SGD) -- minibatch_size = 1
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)
for i in range(0, num_iterations):
    for j in range(0, m):
        # Forward propagation
        a, caches = forward_propagation(X[:,j], parameters)
        # Compute cost
        cost = compute_cost(a, Y[:,j])
        # Backward propagation
        grads = backward_propagation(a, caches, parameters)
        # Update parameters.
        parameters = update_parameters(parameters, grads)
"""

What you should remember: 
- The difference between gradient descent, 
  mini-batch gradient descent and stochastic gradient descent is 
  the number of examples you use to perform one update step. 
- You have to tune a learning rate hyperparameter α. 
- With a well-turned mini-batch size, usually it outperforms either 
  gradient descent or stochastic gradient descent (particularly when 
  the training set is large).
"""
    
#%%===============================2 - Mini-Batch Gradient descent
def random_mini_batches(X,Y,mini_batch_size = 64, seed = 0):
    """
    Creates a list of random minibatches from (X, Y)
    Returns:
    mini_batches -- list of synchronous (mini_batch_X, mini_batch_Y)
    """
    np.random.seed(seed)
    m =X.shape[1]
    mini_batches = []

    permutation = list(np.random.permutation(m)) #“洗牌”
    shuffled_X = X[:,permutation]
    shuffled_Y = Y[:,permutation].reshape((1,m))
    
    num_complte_minibatches = math.floor(m/mini_batch_size)#注意：这里并不包括最后一个minibatch（因为可能最后一块并没有mini_batch_size个数据）
    for k in range(0,num_complte_minibatches):
        mini_batch_X = shuffled_X[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch_Y = shuffled_Y[:,k*mini_batch_size:(k+1)*mini_batch_size]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
    
    if m % mini_batch_size !=0:
        mini_batch_X = shuffled_X[:,num_complte_minibatches*mini_batch_size:m]
        mini_batch_Y = shuffled_Y[:,num_complte_minibatches*mini_batch_size:m]
        mini_batch = (mini_batch_X,mini_batch_Y)
        mini_batches.append(mini_batch)
        
    return mini_batches
"""    
X_assess, Y_assess, mini_batch_size = random_mini_batches_test_case()
mini_batches = random_mini_batches(X_assess, Y_assess, mini_batch_size)#看一下mini_batches的数据结构

print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape)) 
print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))
"""
    
"""note:
- Shuffling and Partitioning are the two steps required to build mini-batches 
- Powers of two are often chosen to be the mini-batch size, e.g., 16, 32, 64, 128.
"""
X = data_input
Y = labels
parameters = initialize_parameters(layers_dims)

mini_batches = random_mini_batches(X,Y,mini_batch_size = 64, seed = 0)
T = len(mini_batches)

for i in range(0, num_iterations):
    for t in range(0, T):
        # Forward propagation
        a, caches = forward_propagation(mini_batches[t][0], parameters)
        # Compute cost
        cost = compute_cost(a, mini_batches[t][1])
        # Backward propagation
        grads = backward_propagation(a, caches, parameters)
        # Update parameters.
        parameters = update_parameters(parameters, grads)

#%%================================3 - Momentum
"""
-动量梯度下降的基本思想就是计算梯度的指数加权平均数，并利用该梯度来更新权重。
-目的是：希望在偏离径方向的梯度下降的慢一些，不要有太大的波动；而沿径向的梯度下降的快一些，使得能够更快的到达最小值点
-采用指数加权平均数，原来在偏离径向的上下波动，经过平均之后，接近于0；而在径向方向上，所有的微分都指向径向，平均值自然较大
-本质解释：将Cost function想象为一个碗状，想象从顶部往下滚球，其中
          微分dw,db想象为球提供的加速度
          动量vdw，vdb想象为速度
          小球在向下滚动的过程中，因为加速度的存在速度会变快，但是由于ββ的存在，其值小于1，可以认为是摩擦力，所以球不会无限加速下去
"""
def initialize_velocity(parameters):
    L = len(parameters)//2
    v = {}

    for l in range(L):
        v["dW"+str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db"+str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        
    return v
    
"""
parameters = initialize_velocity_test_case()
v = initialize_velocity(parameters)
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
"""
def update_parameters_with_momentum(parameters,grads,v,beta ,learning_rate):
    L = len(parameters)//2
    for l in range(L):
        v["dW"+str(l+1)] = beta*v["dW"+str(l+1)]+(1-beta)*grads["dW"+str(l+1)]
        v["db"+str(l+1)] = beta*v["db"+str(l+1)]+(1-beta)*grads["db"+str(l+1)]
          
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*v["dW"+str(l+1)]
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*v["db"+str(l+1)]    
    
    return parameters,v

"""
parameters, grads, v = update_parameters_with_momentum_test_case()

parameters, v = update_parameters_with_momentum(parameters, grads, v, beta = 0.9, learning_rate = 0.01)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
"""

"""
- beta = 0,then this just becomes standard gradient descent without momentum.
- The larger the momentum β is, the smoother the update because the more we take the past gradients into account. But if β is too big, it could also smooth out the updates too much.
- Common values for β range from 0.8 to 0.999. If you don’t feel inclined to tune this, β=0.9β=0.9 is often a reasonable default.
- Tuning the optimal ββ for your model might need trying several values to see what works best in term of reducing the value of the cost function JJ.
"""

#tip:可以加入到 batch gradient descent, mini-batch gradient descent or stochastic gradient descent

#%%===================================4 - Adam
"""
RMSprop
将微分项进行平方，
然后使用平方根进行梯度更新，
同时为了确保算法不会除以0，平方根分母中在实际使用会加入一个很小的值如ε=10−8

Adam:
    Momentum 和 RMSprop 
"""

def initialize_adam(parameters):
    L = len(parameters)//2
    v = {}
    s = {}

    for l in range(L):
        v["dW"+str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        v["db"+str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        s["dW"+str(l+1)] = np.zeros(parameters["W"+str(l+1)].shape)
        s["db"+str(l+1)] = np.zeros(parameters["b"+str(l+1)].shape)
        
    return v,s
"""
parameters = initialize_adam_test_case()

v, s = initialize_adam(parameters)
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
print("s[\"dW1\"] = " + str(s["dW1"]))
print("s[\"db1\"] = " + str(s["db1"]))
print("s[\"dW2\"] = " + str(s["dW2"]))
print("s[\"db2\"] = " + str(s["db2"]))
"""

def update_parameters_with_adam(parameters,grads,v,s,t,learning_rate = 0.01,
                                beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
    L = len(parameters)//2
    v_corrected = {}
    s_corrected = {}

    for l in range(L):
        #Momentum
        v["dW"+str(l+1)] = beta1*v["dW"+str(l+1)] + (1-beta1)*grads["dW"+str(l+1)]
        v["db"+str(l+1)] = beta1*v["db"+str(l+1)] + (1-beta1)*grads["db"+str(l+1)]

        #RMSprop
        s["dW"+str(l+1)] = beta2*s["dW"+str(l+1)] + (1-beta2)*((grads["dW"+str(l+1)])**2)
        s["db"+str(l+1)] = beta2*s["db"+str(l+1)] + (1-beta2)*((grads["db"+str(l+1)])**2)
        
        #bias corrected
        v_corrected["dW"+str(l+1)] = v["dW"+str(l+1)]/(1-beta1**t)
        v_corrected["db"+str(l+1)] = v["db"+str(l+1)]/(1-beta1**t)
        s_corrected["dW"+str(l+1)] = s["dW"+str(l+1)]/(1-beta2**t)
        s_corrected["db"+str(l+1)] = s["db"+str(l+1)]/(1-beta2**t)
        
        parameters["W"+str(l+1)] = parameters["W"+str(l+1)] - learning_rate*v_corrected["dW"+str(l+1)]/(np.sqrt(s_corrected["dW"+str(l+1)])+epsilon)
        parameters["b"+str(l+1)] = parameters["b"+str(l+1)] - learning_rate*v_corrected["db"+str(l+1)]/(np.sqrt(s_corrected["db"+str(l+1)])+epsilon)

    return parameters,v,s
"""    
parameters, grads, v, s = update_parameters_with_adam_test_case()
parameters, v, s  = update_parameters_with_adam(parameters, grads, v, s, t = 2)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
print("v[\"dW1\"] = " + str(v["dW1"]))
print("v[\"db1\"] = " + str(v["db1"]))
print("v[\"dW2\"] = " + str(v["dW2"]))
print("v[\"db2\"] = " + str(v["db2"]))
print("s[\"dW1\"] = " + str(s["dW1"]))
print("s[\"db1\"] = " + str(s["db1"]))
print("s[\"dW2\"] = " + str(s["dW2"]))
print("s[\"db2\"] = " + str(s["db2"]))
"""

#%%============================5 - Model with different optimization algorithms
train_X, train_Y = load_dataset()

def model(X,Y,layers_dims,optimizer,learning_rate = 0.0007,mini_batch_size = 64, beta = 0.9,
          beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8, num_epochs = 10000,print_cost = True):
    L = len(layers_dims)
    costs = []
    t = 0
    seed = 10
    
    parameters = initialize_parameters(layers_dims)
    
    if optimizer == "gd":
        pass
    elif optimizer == "momentum":
        v = initialize_velocity(parameters)
    elif optimizer == "adam":
        v,s = initialize_adam(parameters)
        
    for i in range(num_epochs):#{
        seed = seed + 1
        minibatches = random_mini_batches(X,Y,mini_batch_size,seed)
        
        for minibatch in minibatches:#{
            (minibatch_X,minibatch_Y) = minibatch
            
            a3,caches = forward_propagation(minibatch_X,parameters)
            cost = compute_cost(a3,minibatch_Y)
            grads = backward_propagation(minibatch_X,minibatch_Y,caches)
        
            if optimizer == "gd":
                parameters = update_parameters_with_gd(parameters,grads,learning_rate)
            elif optimizer == "momentum":
                parameters,v == update_parameters_with_momentum(parameters,grads,v,beta,learning_rate)
            elif optimizer == "adam":
                t = t + 1 #时间量，用于Adam更新参数前，偏差修正
                parameters,v,s == update_parameters_with_adam(parameters,grads,v,s,t,learning_rate,beta1,beta2,epsilon)
        #}    
        if print_cost and i%1000 == 0:
            print("Cost after epoch %i:%f"%(i,cost))
        if print_cost and i%100 == 0 :
            costs.append(cost)
    #}
    # plot the cost
    plt.plot(costs)
    plt.ylabel('cost')
    plt.xlabel('epochs (per 100)')
    plt.title("Learning rate = " + str(learning_rate))
    plt.show()
        
    return parameters    
        
#%%-------------------------------5.1 Mini-batch Gradient descent        
layers_dims = [train_X.shape[0],5,2,1]
parameters = model(train_X,train_Y,layers_dims,optimizer = "gd")

predictions = predict(train_X,train_Y,parameters)

# Plot decision boundary
plt.title("Model with Gradient Descent optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)       
'''
Accuracy = 0.796666666667
'''        

#%%----------------------------5.2 Mini-batch gradient descent with momentum
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, beta = 0.9, optimizer = "momentum")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Momentum optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
'''
Accuracy = 0.796666666667
'''  
#%%------------------------------5.3 Mini-batch with Adam mode
layers_dims = [train_X.shape[0], 5, 2, 1]
parameters = model(train_X, train_Y, layers_dims, optimizer = "adam")

# Predict
predictions = predict(train_X, train_Y, parameters)

# Plot decision boundary
plt.title("Model with Adam optimization")
axes = plt.gca()
axes.set_xlim([-1.5,2.5])
axes.set_ylim([-1,1.5])
plot_decision_boundary(lambda x: predict_dec(parameters, x.T), train_X, train_Y)
'''
Accuracy: 0.936666666667
'''

"""
- Adam clearly outperforms mini-batch gradient descent and Momentum. 
  If you run the model for more epochs on this simple dataset, 
  all three methods will lead to very good results.
  However, you’ve seen that Adam converges a lot faster.
  
  Some advantages of Adam include: 
- Relatively low memory requirements (though higher than gradient descent and gradient descent with momentum) 
- Usually works well even with little tuning of hyperparameters (except α)
"""

#%%================================6. learning_rate decay
#学习率衰减



