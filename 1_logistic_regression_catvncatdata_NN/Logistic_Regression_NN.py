# -*- coding: utf-8 -*-
"""
Created on Thu Jul 12 11:22:02 2018

@author: a
"""


#==================================Logistic Regression with a Neural Network mindset 
"""
You will learn to: 
- Build the general architecture of a learning algorithm, including: 
- Initializing parameters 
- Calculating the cost function and its gradient 
- Using an optimization algorithm (gradient descent) 
- Gather all three functions above into a main model function, in the right order.
"""
#---------------------Packages
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset
                    
#----------------------Overview of the Problem set
# Loading the data (cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()                    
"""
# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") +  "' picture.")
"""                    
                    
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]                    
                    
print ("Number of training examples: m_train = " + str(m_train))  #209
print ("Number of testing examples: m_test = " + str(m_test))     #50
print ("Height/Width of each image: num_px = " + str(num_px))     #64
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
                    
#Reshape the training and test data sets so that images of size (num_px, num_px, 3) 
#are flattened into single vectors of shape (num_px ∗∗ num_px ∗∗ 3, 1).
train_set_x_flatten = train_set_x_orig.reshape(m_train, -1).T
test_set_x_flatten = test_set_x_orig.reshape(m_test, -1).T                    
    

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_set_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("sanity check after reshaping: " + str(train_set_x_flatten[0:5,0]))
    
#standardize
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.    

#NOTE
"""
Common steps for pre-processing a new dataset are: 
- Figure out the dimensions and shapes of the problem (m_train, m_test, num_px, …) 
- Reshape the datasets such that each example is now a vector of size (num_px * num_px * 3, 1) 
- “Standardize” the data
"""
def sigmoid(z):
    s = 1.0/(1+np.exp(-z))
    return s
    
#------------------------------Initializing parameters 初始化参数
def initialize_with_zeros(dim):
    """
    Argument:
    dim  -- size of the w vector    
    
    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """
    
    w = np.zeros((dim,1))
    b = 0
    
    assert(w.shape == (dim,1))
    assert(isinstance(b,float) or isinstance(b,int))
    
    return w,b
    
"""#example
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))
"""
    
#---------------------------Forward and Backward propagation 正向传播和反向传播

def propagate(w,b,X,Y):
    """
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)

    Return:
    cost -- negative log-likelihood cost for logistic regression
    dw -- gradient of the loss with respect to w, thus same shape as w
    db -- gradient of the loss with respect to b, thus same shape as b
    """
    
    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T,X)+b)
    cost = -(1.0/m)*np.sum(Y*np.log(A)+(1-Y)*np.log(1-A))
    
    # BACKWARD PROPAGATION (TO FIND GRAD)
    dw = (1.0/m)*np.dot(X,(A-Y).T)
    db = (1.0/m)*np.sum(A-Y)
    
    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    
    grads = {"dw": dw,
             "db": db}
    return grads , cost
    
#-----------------------Optimization
#update the parameters using Gradient Descent.

def optimize(w,b,X,Y,num_iterations,learning_rate,print_cost = False):
    #num_iterations --更新循环次数
    #learning_rate  -- 学习率
    #print_cost     --True to print the loss every 100 steps

    costs = []
    
    for i in range(num_iterations):
        
        grads,cost = propagate(w,b,X,Y)
        
        dw = grads["dw"]
        db = grads["db"]

        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        if i%100 == 0:
            costs.append(cost)
            
        if print_cost and i%100 ==0:
            print("Cost after iteration %i: %f" %(i, cost))
    #for end
        
    params = {"w": w,
              "b": b}

    grads = {"dw": dw,
             "db": db}

    return params, grads, costs
    
#------------------------------Predict

def predict(w,b,X):
    
    m = X.shape[1]
    Y_predition = np.zeros((1,m))
    w = w.reshape(X.shape[0],1)
    
    A = sigmoid(np.dot(w.T,X)+b)
    
    for i in range(A.shape[1]):
        if A[0,i] > 0.5:
            Y_predition[0,i] = 1
        else:
            Y_predition[0,i] = 0

    assert(Y_predition.shape == (1,m))
    
    return Y_predition

#----------------------Merge all functions into a model

def model(X_train,Y_train,X_test,Y_test,num_iterations = 2000,learning_rate = 0.5,print_cost = False):
    
    w,b = initialize_with_zeros(X_train.shape[0])
    parameters,grads,costs = optimize(w,b,X_train,Y_train,num_iterations,learning_rate,print_cost)
    
    w = parameters["w"]
    b = parameters["b"]

    Y_prediction_train = predict(w,b,X_train)
    Y_prediction_test = predict(w,b,X_test)
    
    # Print train/test Errors
    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test, 
         "Y_prediction_train" : Y_prediction_train, 
         "w" : w, 
         "b" : b,
         "learning_rate" : learning_rate,
         "num_iterations": num_iterations}

    return d
    
#============================Run
d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations = 2000, learning_rate = 0.005, print_cost = True)

#-------------test one of data
# Example of a picture that was wrongly classified.

index = 1
plt.imshow(test_set_x[:,index].reshape((num_px, num_px, 3)))
print ("y = " + str(test_set_y[0,index]) + ", you predicted that it is a \"" + classes[d["Y_prediction_test"][0,index]].decode("utf-8") +  "\" picture.")

#---------Plot learning curve (with costs)
costs = np.squeeze(d['costs'])
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per hundreds)')
plt.title("Learning rate =" + str(d["learning_rate"]))
plt.show()

#-----------Different learning_rate plot

learning_rates = [0.01,0.001,0.0001]
models = {}
for i in learning_rates:
    print("learning_rate is :" + str(i))
    models[str(i)] = model(train_set_x,train_set_y,test_set_x,test_set_y,num_iterations = 1500 , learning_rate = i,print_cost = False)
    print('\n'+"-----------------------------------------"+'\n')
    
for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]),label = str(models[str(i)]["learning_rate"]))
    
plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc = 'upper center',shadow = True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.show()







