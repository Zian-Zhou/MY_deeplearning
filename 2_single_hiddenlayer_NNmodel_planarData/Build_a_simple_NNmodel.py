# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 20:30:47 2018

@author: a
"""

#====================Planar data classification with one hidden layer

#----------------package
import numpy as np
import matplotlib.pyplot as plt
from testCases_v2 import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

%matplotlib inline

np.random.seed(1) # set a seed so that the results are consistent

#-----------------DataSet

def load_planar_dataset():
    np.random.seed(1)
    m = 400 #样本数量
    N = int(m/2) #每个类别的样本量
    D = 2 #维度数
    X = np.zeros((m,D))
    Y = np.zeros((m,1),dtype='uint8')
    a = 4 #花的最大长度
    
    for j in range(2):
        ix = range(N*j,N*(j+1))
        t = np.linspace(j*3.12,(j+1)*3.12,N)+np.random.randn(N)*0.2 #
        r = a*np.sin(4*t) + np.random.randn(N)*0.2 #radius
        X[ix] = np.c_[r*np.sin(t),r*np.cos(t)] #
        Y[ix] = j

    X = X.T
    Y = Y.T
    
    return X,Y

X,Y = load_planar_dataset()

#Visualize the data:数据可视化
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);

#------------------The shape of X,Y
shape_X = X.shape
shape_Y = Y.shape
m = X.shape[1]

print('The shape of X is:'+str(shape_X))
print('The shape of Y is:'+str(shape_Y))
print('Here is m = %d training examples.'%(m))

#--------------------1、Simple Logistic Regression
'''
Before building a full neural network, 
lets first see how logistic regression performs on this problem. 
You can use sklearn’s built-in functions to do that. 
Run the code below to train a logistic regression classifier on the dataset.
'''
clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T,Y.T)

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)

# Plot the decision boundary for logistic regression
plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")

# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
       '% ' + "(percentage of correctly labelled datapoints)")    

'''
显然，由于数据集线性不可分，LOGISTIC回归不能很好的处理这样的数据集（线性函数的组合）：注意理解这句话的含义  回到原理本身  为什么是线性？
'''

#=---------------------------2.Neural Network model
#Define NN structure
'''
n_x:the size of input layer
n_h:the size of hidden layer(set 4)
n_y:the size of the output layer
'''
def layer_sizes(X,Y):
    '''
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    '''
    
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    
    return (n_x,n_h,n_y)

'''#example
X_assess, Y_assess = layer_sizes_test_case()
(n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))
'''

#Initialize the model's parameters
def initialize_parameters(n_x,n_h,n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer

    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    np.random.seed(2)
    
    W1 = np.random.randn(n_h,n_x)*0.01
    b1 = np.zeros((n_h,1))
    W2 = np.random.randn(n_y,n_h)*0.01
    b2 = np.zeros((n_y,1))
    
    assert (W1.shape == (n_h,n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))    
    
    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2
                  }
    
    return parameters

'''#example
n_x, n_h, n_y = initialize_parameters_test_case()

parameters = initialize_parameters(n_x, n_h, n_y)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''
    
#LOOP

def forward_propagation(X,parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)

    Returns:
    A2 -- The sigmoid output of the second activation
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"#缓冲池，BP计算时用到
    """
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    Z1 = np.dot(W1,X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + b2
    A2 = sigmoid(Z2)
    
    assert (A2.shape == (1,X.shape[1]))
    
    cache = {"Z1":Z1,
             "A1":A1,
             "Z2":Z2,
             "A2":A2}
            
    return A2, cache
    
'''
X_assess, parameters = forward_propagation_test_case()
A2, cache = forward_propagation(X_assess, parameters)

# Note: we use the mean here just to make sure that your output matches ours. 
print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))
'''

def compute_cost(A2,Y,paramters):
    
    m = Y.shape[1]

    logprobs = np.multiply(Y,np.log(A2)) + np.multiply((1-Y),np.log(1-A2))
    cost = -(1.0/m)*np.sum(logprobs)
    
    cost = np.squeeze(cost) # makes sure cost is the dimension we expect. 
                            # E.g., turns [[17]] into 17 
    assert (isinstance(cost,float))
    
    return cost
    
'''#example
A2, Y_assess, parameters = compute_cost_test_case()

print("cost = " + str(compute_cost(A2, Y_assess, parameters)))
'''

def backward_propagation(parameters, cache, X, Y):
    '''
    Returns:
    grads -- python dictionary containing your gradients with respect to different parameters
    '''
    
    m = X.shape[1]

    W1 = parameters["W1"]
    W2 = parameters["W2"]

    A1 = cache["A1"]
    A2 = cache["A2"]

    dZ2 = A2 - Y
    dW2 = 1.0/m*np.dot(dZ2, A1.T)
    db2 = 1.0/m*np.sum(dZ2, axis=1, keepdims = True)    
    dZ1 = np.dot(W2.T, dZ2)*(1-np.power(A1,2))#?
    '''
    由于激活函数为sigmoid，可以推导得到dZ2  = A2 - Y
    由于激活函数为a = tanh(z)，导函数为1-a^2,即(1-np.power(A1,2))
    '''
    dW1 = 1.0/m*np.dot(dZ1, X.T)
    db1 = 1.0/m*np.sum(dZ1, axis=1, keepdims = True)
    
    grads = {"dW1":dW1,
             "db1":db1,
             "dW2":dW2,
             "db2":db2}
            
    return grads
    
'''#example
parameters, cache, X_assess, Y_assess = backward_propagation_test_case()

grads = backward_propagation(parameters, cache, X_assess, Y_assess)
print ("dW1 = "+ str(grads["dW1"]))
print ("db1 = "+ str(grads["db1"]))
print ("dW2 = "+ str(grads["dW2"]))
print ("db2 = "+ str(grads["db2"]))
'''
    
def update_parameters(parameters, grads, learning_rate = 1.2):
    
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    W1 = W1 - learning_rate*dW1
    b1 = b1 - learning_rate*db1
    W2 = W2 - learning_rate*dW2
    b2 = b2 - learning_rate*db2
    
    parameters = {"W1":W1,
                  "b1":b1,
                  "W2":W2,
                  "b2":b2}
                 
    return parameters
    
'''#example
parameters, grads = update_parameters_test_case()
parameters = update_parameters(parameters, grads)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''

#Build NN model
def nn_model(X,Y,n_h,num_iterations =1000,print_cost = False):
    np.random.seed(3)
    n_x = layer_sizes(X,Y)[0]
    n_y = layer_sizes(X,Y)[2]

    parameters = initialize_parameters(n_x,n_h,n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    for i in range(num_iterations):
    #{
        A2,cache = forward_propagation(X,parameters)
        
        cost = compute_cost(A2,Y,parameters)
        
        grads = backward_propagation(parameters,cache,X,Y)
        
        parameters = update_parameters(parameters,grads,learning_rate = 1.2)
        
        if print_cost and i%1000 == 0:
            print("Cost after iteration %i: %f"%(i,cost))
    #}        
    
    return parameters

'''
X_assess, Y_assess = nn_model_test_case()
parameters = nn_model(X_assess, Y_assess, 4, num_iterations=10000, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
'''
        
        
#-----------------------------3.Prediction

def predict(paramters, X, threshold = 0.5):
    '''
    Using the learned parameters, predicts a class for each example in X

    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    threshold
    
    Returns:
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    '''
    A2,cache = forward_propagation(X,paramters)
    predictions = (A2 > threshold)
    
    return predictions
    
'''#example
parameters, X_assess = predict_test_case()
predictions = predict(parameters, X_assess)
print("predictions mean = " + str(np.mean(predictions)))
'''
    
#--------------------------Test with the planar dataset,using a single hidden layer of n_h hidden units
parameters = nn_model(X, Y, n_h = 4, num_iterations = 10000, print_cost = True)

plot_decision_boundary(lambda x: predict(parameters, x.T, threshold=0.5),X,Y)
plt.title("Decision Boundary for hidden layer size " + str(4))

predictions = predict(parameters,X)
print('Accuracy: %d'%float((np.dot(Y,predictions.T)+np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

#--------------------------Turning hidden layer size
'''
测试不同隐层节点数的模型
'''
#take some time
plt.figure(figsize=(16,32))
hidden_layer_sizes = [1,2,3,4,5,20,50]
for i,n_h in enumerate(hidden_layer_sizes):
    plt.subplot(5,2,i+1)
    plt.title('Hidden Layer of size %d'% n_h)
    parameters = nn_model(X,Y,n_h,num_iterations = 5000)
    plot_decision_boundary(lambda x:predict(parameters,x.T),X,Y)
    predictions = predict(parameters,X)
    accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
    print("Accuracy for {} hidden units: {} %".format(n_h,accuracy))

'''
You’ve learnt to: 
- Build a complete neural network with a hidden layer 
- Make a good use of a non-linear unit 
- Implemented forward propagation and backpropagation, and trained a neural network 
- See the impact of varying the hidden layer size, including overfitting.
'''
#=================Performance on other datasets
'''
以下是新数据集，自己训练模型（未完成）
'''
def load_extra_datasets():  
    N = 200
    noisy_circles = sklearn.datasets.make_circles(n_samples=N, factor=.5, noise=.3)
    noisy_moons = sklearn.datasets.make_moons(n_samples=N, noise=.2)
    blobs = sklearn.datasets.make_blobs(n_samples=N, random_state=5, n_features=2, centers=6)
    gaussian_quantiles = sklearn.datasets.make_gaussian_quantiles(mean=None, cov=0.5, n_samples=N, n_features=2, n_classes=2, shuffle=True, random_state=None)
    no_structure = np.random.rand(N, 2), np.random.rand(N, 2)

    return noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure
# Datasets
noisy_circles, noisy_moons, blobs, gaussian_quantiles, no_structure = load_extra_datasets()

datasets = {"noisy_circles": noisy_circles,
            "noisy_moons": noisy_moons,
            "blobs": blobs,
            "gaussian_quantiles": gaussian_quantiles}

### START CODE HERE ### (choose your dataset)
dataset = "noisy_moons"
### END CODE HERE ###

X, Y = datasets[dataset]
X, Y = X.T, Y.reshape(1, Y.shape[0])

# make blobs binary
if dataset == "blobs":
    Y = Y%2

# Visualize the data
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral);