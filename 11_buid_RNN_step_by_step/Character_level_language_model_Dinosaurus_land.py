# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 13:47:43 2018

@author: a
"""
#%%================Part 2 - Character level language model - Dinosaurus land

#%%
import numpy as np
from utils import *
import random

#%%````````````````````````1 - Problem Statement
#1.1 - Dataset and Preprocessing
data = open('dinos.txt', 'r').read()
data= data.lower()
chars = list(set(data))
data_size, vocab_size = len(data), len(chars)
#print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))

char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
print(ix_to_char)

#%%`````````````````````````2 - Building blocks of the model

#2.1 - Clipping the gradients in the optimization loop
"""
In this section you will implement the clip function that you will call inside of your optimization loop. 
Recall that your overall loop structure usually consists of a forward pass, a cost computation, a backward pass, and a parameter update. 
Before updating the parameters, you will perform gradient clipping when needed to make sure that your gradients are not “exploding,” meaning taking on overly large values.

In the exercise below, you will implement a function clip that takes in a dictionary of gradients and returns a clipped version of gradients if needed. 
There are different ways to clip gradients; we will use a simple element-wise clipping procedure, in which every element of the gradient vector is clipped to lie between some range [-N, N]. 
More generally, you will provide a maxValue (say 10). 
In this example, if any component of the gradient vector is greater than 10, it would be set to 10; 
and if any component of the gradient vector is less than -10, it would be set to -10. 
If it is between -10 and 10, it is left alone.
"""
def clip(gradients,maxValue):
    dWaa, dWax, dWya, db, dby = gradients['dWaa'],gradients['dWax'],gradients['dWya'],gradients['db'],gradients['dby']
    
    for gradients in [dWaa,dWax,dWya,db,dby]:
        np.clip(gradients, -maxValue, maxValue, out = gradients)

    gradients = {"dWaa": dWaa, "dWax": dWax, "dWya": dWya, "db": db, "dby": dby}

    return gradients
"""
np.random.seed(3)
dWax = np.random.randn(5,3)*10
dWaa = np.random.randn(5,5)*10
dWya = np.random.randn(2,5)*10
db = np.random.randn(5,1)*10
dby = np.random.randn(2,1)*10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients = clip(gradients, 10)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
"""

#2.2 - Sampling
"""
看图：sample.png
采用np.random.choice,就会根据预先给好概率分布来采样
"""
def sample(parameters,char_to_ix,seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0]
    n_a = Waa.shape[1]
    
    x = np.zeros((vocab_size,1))
    a_prev = np.zeros((n_a,1))#a0
    
    indices = []
    
    # Idx is a flag to detect a newline character, we initialize it to -1
    idx = -1
    
    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append 
    # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well 
    # trained model), which helps debugging and prevents entering an infinite loop. 
    counter = 0
    newline_character = char_to_ix['\n']
    
    while (idx != newline_character and counter != 50):
        a = np.tanh(np.dot(Wax,x) + np.dot(Waa,a_prev) + b)
        z = np.dot(Wya, a) + by
        y = softmax(z)
        
        np.random.seed(counter+seed)
        
        idx = np.random.choice(range(len(y)), p = y.ravel())
        
        indices.append(idx)
        
        x = np.zeros((vocab_size,1))
        x[idx] = 1
        
        a_prev = a
        
        seed += 1
        counter += 1
        
    if counter==50:
        indices.append(newline_character)
        
    return indices
"""
np.random.seed(2)
_, n_a = 20, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


indices = sample(parameters, char_to_ix, 0)
print("Sampling:")
print("list of sampled indices:", indices)
print("list of sampled characters:", [ix_to_char[i] for i in indices])
"""

#%%`````````````````````````````````3 - Building the language model
#3.1 - Gradient descent
def optimize(X,Y,a_prev,parameters,learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.

    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.

    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    loss, cache = rnn_forward(X,Y,a_prev,parameters)
    gradients, a = rnn_backward(X,Y,parameters,cache)
    
    gradients = clip(gradients,5)
    
    parameters = update_parameters(parameters,gradients,learning_rate)
    
    return loss, gradients, a[len(X)-1]

"""
np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])
"""

#3.2 - Training the model
def model(data,ix_to_char,char_to_ix,num_iterations=35000,n_a=50,dino_names=7,vocab_size=27):
    """
    Trains the model and generates dinosaur names. 

    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration. 
    vocab_size -- number of unique characters found in the text, size of the vocabulary

    Returns:
    parameters -- learned parameters
    """
    n_x,n_y = vocab_size,vocab_size
    parameters = initialize_parameters(n_a,n_x,n_y)
    loss = get_initial_loss(vocab_size,dino_names)
    
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]#x.lower():改成小写，.strip()：去掉'\n'
    
    np.random.seed(0)
    np.random.shuffle(examples)
    
    a_prev = np.zeros((n_a,1))
    
    for j in range(num_iterations):
        index = j%len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]]
        Y = X[1:] + [char_to_ix["\n"]]
        """
        Note that we use: index= j % len(examples), where j = 1....num_iterations, to make sure that examples[index] is always a valid statement (index is smaller than len(examples)). 
        The first entry of X being None will be interpreted by rnn_forward() as setting x⟨0⟩=0⃗ . Further, this ensures that Y is equal to X but shifted one step to the left, and with an additional “\n” appended to signify the end of the dinosaur name.
        """
        curr_loss,gradients,a_prev = optimize(X,Y,a_prev,parameters,learning_rate=0.01)
        
        loss = smooth(loss,curr_loss)## Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        
        if j%2000 == 0:
            print('Iteration: %d, Loss: %f'%(j,loss)+'\n')
            
            seed = 0
            for name in range(dino_names):
                sample_indices = sample(parameters,char_to_ix,seed)
                print_sample(sample_indices,ix_to_char)
                seed = seed+1
                
            print('\n')
            
    return parameters
                
#parameters = model(data, ix_to_char, char_to_ix)



















