# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 17:03:02 2018

@author: a
"""
#%%===========================Improvise a Jazz Solo with an LSTM Network     with  Keras


from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K

#%%``````````````````````````````1 - Problem statement
IPython.display.Audio('./data/30s_seq.mp3')

X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)

""" 
X: This is an (m, Tx, 78) dimensional array. We have m training examples, each of which is a snippet of Tx=30 musical values. At each time step, the input is one of 78 different possible values, represented as a one-hot vector. Thus for example, X[i,t,:] is a one-hot vector representating the value of the i-th example at time t.

Y: This is essentially the same as X, but shifted one step to the left (to the past). Similar to the dinosaurus assignment, we’re interested in the network using the previous values to predict the next value, so our sequence model will try to predict y⟨t⟩y⟨t⟩ given x⟨1⟩,…,x⟨t⟩x⟨1⟩,…,x⟨t⟩. However, the data in Y is reordered to be dimension (Ty,m,78), where Ty=Tx. This format makes it more convenient to feed to the LSTM later.

n_values: The number of unique values in this dataset. This should be 78.

indices_values: python dictionary mapping from 0-77 to musical values.
"""

#the model is shown on "Jazz lstm model.png"

#%%``````````````````````````````2 - Building the model
"""
In this part you will build and train a model that will learn musical patterns. 
To do so, you will need to build a model that takes in X of shape (m,Tx,78) and Y of shape (Ty,m,78). 
We will use an LSTM with 64 dimensional hidden states. Lets set n_a = 64.
"""
n_a = 64

"""# Implement djmodel().
1.Create an empty list “outputs” to save the outputs of the LSTM Cell at every time step.
2.Loop for t∈1,…,Txt∈1,…,Tx:A,B,C,D,E
"""

#B. Reshape x to be (1,78). You may find the `reshapor()` layer (defined below) helpful.
#reshapor = Reshape((1,78))

##C. Run x through one step of LSTM_cell. Remember to initialize the LSTM_cell with the previous step's hidden state $a$ and cell state $c$. Use the following formatting:
##   a, _, c = LSTM_cell(input_x, initial_state=[previous hidden state, previous cell state])
#LSTM_cell = LSTM(n_a,return_state = True)

##D. Propagate the LSTM's output activation value through a dense+softmax layer using `densor`. 
#densor = Dense(n_values,activation='softmax')

"""
  Each of reshapor, LSTM_cell and densor are now layer objects,
and you can use them to implement djmodel(). 
  In order to propagate a Keras tensor object X through one of these layers, 
use layer_object(X) (or layer_object([X,Y]) if it requires multiple inputs.). 
  For example, reshapor(X) will propagate X through the Reshape((1,78)) layer defined above.
"""
## A.Select the “t”th time-step vector from X. The shape of this selection should be (78,). To do so, create a custom Lambda layer in Keras by using this line of code:
#x = Lambda(lambda x:X[:,t,:])(X)

## E.Append the predicted value to the list of "outputs"


reshapor = Reshape((1,78))
LSTM_cell = LSTM(n_a,return_state = True)
densor = Dense(n_values,activation='softmax')

def djmodel(Tx,n_a,n_values):
    """
    Implement the model

    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 

    Returns:
    model -- a keras model with the 
    """
    X = Input(shape=(Tx,n_values))
    
    a0 = Input(shape=(n_a,),name='a0')
    c0 = Input(shape=(n_a,),name='c0')
    a = a0
    c = c0

    #step1:create empty list to append the outputs while iterate
    outputs = []
    
    #step2: LOOP
    for t in range(Tx):
        #A
        x = Lambda(lambda x:X[:,t,:])(X)
        #B
        x = reshapor(x)
        #C
        a,_,c = LSTM_cell(x,initial_state=[a,c])
        #D
        out = densor(a)
        #E
        outputs.append(out)
        
    model = Model(inputs=[X,a0,c0],outputs=outputs)
    
    return model

model = djmodel(Tx = 30 , n_a = 64, n_values = 78)

opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01)

model.compile(optimizer=opt, loss='categorical_crossentropy',metrics=['accuracy'])

#initialize a0 and c0 for the LSTM’s initial state to be zero.
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))

#train
model.fit([X,a0,c0],list(Y),epochs=100)


#%%``````````````````````````3 - Generating music
#3.1 - Predicting & Sampling
def music_inference_model(LSTM_cell,densor,n_values=78,n_a=64,Ty=100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.

    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate

    Returns:
    inference_model -- Keras model instance
    """
    x0 = Input(shape=(1,n_values))
    
    a0 = Input(shape=(n_a,),name='a0')
    c0 = Input(shape=(n_a,),name='c0')
    a = a0
    c = c0
    x = x0
    
    #step1
    outputs = []
    
    #step2
    for t in range(Ty):
        #2.A  Perform one step of LSTM_cell
        a,_,c = LSTM_cell(x, initial_state=[a,c])

        #2.B Apply Dense layer to the hidden state output of the LSTM_cell
        out = densor(a)
        
        #2.C Append the prediction "out" to "outputs". out.shape = (None, 78)
        outputs.append(out)
        
        #2.D: Select the next value according to "out", and set "x" to be the one-hot representation of the
        #     selected value, which will be passed as the input to LSTM_cell on the next step. We have provided 
        #     the line of code you need to do this.
        x = Lambda(one_hot)(out)
        
        #step3
        inference_model = Model(inputs=[x0,a0,c0],outputs=outputs)

    return inference_model

inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)


#initialize
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def predict_and_sample(inference_model,x_initializer,a_initializer,c_initializer):
    """
    Predicts the next value of values using the inference model.

    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel

    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer,a_initializer,c_initializer])
    
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
    indices = np.argmax(pred,axis=-1)
    
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (1, )
    results = to_categorical(indices,num_classes=x_initializer.shape[-1])
    
    return results,indices
"""
results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))
#Moreover, you should observe that: np.argmax(results[12]) is the first element of list(indices[12:18]), 
#and np.argmax(results[17]) is the last element of list(indices[12:18]).
"""

#3.3 - Generate music
out_stream = generate_music(inference_model)    #import generate_music from data_utils
    
    
IPython.display.Audio('data/30s_seq_model.mp3')
    
    
#Congratulations!

"""
Here’s what you should remember: 
- A sequence model can be used to generate musical values, which are then post-processed into midi music. 
- Fairly similar models can be used to generate dinosaur names or to generate music, with the major difference being the input fed to the model. 
- In Keras, sequence generation involves defining layers with shared weights, which are then repeated for the different time steps 1,…,Tx1,…,Tx.
"""    
    
    
    
    
    
    
    
