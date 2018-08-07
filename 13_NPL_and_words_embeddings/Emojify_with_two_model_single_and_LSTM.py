# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 15:31:16 2018

@author: a
"""

#%%===========================================Part 2 - Emojify!
import numpy as np
from emo_utils import *
import emoji
import matplotlib.pyplot as plt

%matplotlib inline
##终端输出命令

#%matplotlib qt5
##新窗口输出命令
#%%``````````````````````````````````1 - Baseline model: Emojifier-V1
#------------------------1.1 - Dataset EMOJISET
"""#data set
- X contains 127 sentences (strings) 
- Y contains a integer label between 0 and 4 corresponding to an emoji for each sentence
"""
#load data
X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

maxLen = len(max(X_train, key=len).split())

index = 1
print(X_train[index], label_to_emoji(Y_train[index]))

#-----------------------1.2 - Overview of the Emojifier-V1
#Emojifier-V1.png
"""
To get our labels into a format suitable for training a softmax classifier, 
lets convert YY from its current shape current shape (m,1) into a “one-hot representation” (m,5), 
where each row is a one-hot vector giving the label of one example, 
You can do so using this next code snipper. 
Here, Y_oh stands for “Y-one-hot” in the variable names Y_oh_train and Y_oh_test:
"""
Y_oh_train = convert_to_one_hot(Y_train, C = 5)
Y_oh_test = convert_to_one_hot(Y_test, C = 5)

#-----------------------1.3 - Implementing Emojifier-V1
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
"""
As shown in Figure (2), the first step is to convert an input sentence into the word vector representation, which then get averaged together. Similar to the previous exercise, we will use pretrained 50-dimensional GloVe embeddings.

- word_to_index: dictionary mapping from words to their indices in the vocabulary (400,001 words, with the valid indices ranging from 0 to 400,000) 
- index_to_word: dictionary mapping from indices to their corresponding words in the vocabulary 
- word_to_vec_map: dictionary mapping words to their GloVe vector representation.
"""
#word = "cucumber"
#index = 289846
#print("the index of", word, "in the vocabulary is", word_to_index[word])
#print("the", str(index) + "th word in the vocabulary is", index_to_word[index])

def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """
    words = sentence.lower().split()
    
    avg = np.zeros((50,))
    
    for w in words:
        avg += word_to_vec_map[w]
    avg = avg / len(words)
    
    return avg

"""
avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
print("avg = ", avg)
"""

def model(X,Y,word_to_vec_map,learning_rate = 0.01, num_iterations = 400):
    """
    Model to train word vector representations in numpy.

    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations

    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """
    np.random.seed(1)
    
    m = Y.shape[0]     # number of training examples
    n_y = 5            # number of classes 
    n_h = 50           # dimensions of the GloVe vectors
    
    ## Initialize parameters using Xavier initialization
    W = np.random.randn(n_y,n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))
    
    Y_oh = convert_to_one_hot(Y,C = n_y)
    
    for t in range(num_iterations):
        for i in range(m):#m examples
            avg = sentence_to_avg(X[i],word_to_vec_map)
            
            z = np.dot(W,avg) + b
            a = softmax(z)
            
            cost = - np.sum(Y_oh[i] * np.log(a))
            
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y,1), avg.reshape(1,n_h))
            db = dz
            
            W = W - learning_rate * dW
            b = b - learning_rate * db
            
        if t % 100 == 0:
            print("Epoch: "+str(t)+" --- cost = "+str(cost))
            pred = predict(X,Y,W,b,word_to_vec_map)
            
    return pred,W,b

"""
print(X_train.shape)
print(Y_train.shape)
print(np.eye(5)[Y_train.reshape(-1)].shape)
print(X_train[0])
print(type(X_train))
Y = np.asarray([5,0,0,5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
print(Y.shape)

X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
 'Lets go party and drinks','Congrats on the new job','Congratulations',
 'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
 'You totally deserve this prize', 'Let us go play football',
 'Are you down for football this afternoon', 'Work hard play harder',
 'It is suprising how people can be dumb sometimes',
 'I am very disappointed','It is the best day in my life',
 'I think I will end up alone','My life is so boring','Good job',
 'Great so awesome'])
    
pred, W, b = model(X_train, Y_train, word_to_vec_map)
print(pred)
"""
    
#------------------------------------1.4 - Examining test set performance
print("Training set:")
pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
print('Test set:')
pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)

X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "you are not happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])
#adore不在字典中，但模型仍能给出准确预测，这是因为： Because adore has a similar embedding as love, the algorithm has generalized correctly even to a word it has never seen before. 

pred = predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
print_predictions(X_my_sentences, pred)


print(Y_test.shape)
print('           '+ label_to_emoji(0)+ '    ' + label_to_emoji(1) + '    ' +  label_to_emoji(2)+ '    ' + label_to_emoji(3)+'   ' + label_to_emoji(4))
print(pd.crosstab(Y_test, pred_test.reshape(56,), rownames=['Actual'], colnames=['Predicted'], margins=True))
plot_confusion_matrix(Y_test, pred_test)


"""
What you should remember from this part: 
- Even with a 127 training examples, you can get a reasonably good model for Emojifying. 
  This is due to the generalization power word vectors gives you. 
- Emojify-V1 will perform poorly on sentences such as “This movie is not good and not enjoyable” 
  because it doesn’t understand combinations of words–it just averages all the words’ embedding vectors together, 
  without paying attention to the ordering of words. 
  You will build a better algorithm in the next part.
"""




#%%````````````````````````````````2 - Emojifier-V2: Using LSTMs in Keras:
import numpy as np
np.random.seed(0)
from keras.models import Model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
np.random.seed(1)

from keras.initializers import glorot_uniform

#------------------------------2.1 - Overview of the model
#Emojifier-v2.png

#------------------------------2.2 Keras and mini-batching
"""
most deep learning frameworks require that all sequences in the same mini-batch have the same length. 
so need to pad:
    e_1 = (word11,word12)
    e_2 = (word21,word22,word23)
    maxlen = 5
    -->e1 = (word11,word12,0,0,0)
       e2 = (word21,word22,word23,0,0)
"""
#------------------------------2.3 - The Embedding layer
#emojifier-V2_embedding layer.png
def sentences_to_indices(X,word_to_index,max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4). 

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this. 

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = X.shape[0]
    X_indices = np.zeros((m,max_len))
    
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        
        for w in sentence_words:
            X_indices[i, j] = word_to_index[w]
            j = j + 1
            
    return X_indices

"""
X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
X1_indices = sentences_to_indices(X1,word_to_index, max_len = 5)
print("X1 =", X1)
print("X1_indices =", X1_indices)
"""
def pretrained_embedding_layer(word_to_vec_map,word_to_index):
    vocab_len = len(word_to_index) + 1## adding 1 to fit Keras embedding (requirement)???##
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    
    # Initialize the embedding matrix as a numpy array of zeros of shape (vocab_len, dimensions of word vectors = emb_dim)
    emb_matrix = np.zeros((vocab_len,emb_dim))
    
    # Set each row "index" of the embedding matrix to be the word vector representation of the "index"th word of the vocabulary
    for word, index in word_to_index.items():
        emb_matrix[index,:] = word_to_vec_map[word]

    # Define Keras embedding layer with the correct output/input sizes, make it trainable.
    # Use Embedding(...). Make sure to set trainable=False.
    embedding_layer = Embedding(vocab_len,emb_dim,trainable=False)
    
    # Build the embedding layer, it is required before setting the weights of the embedding layer. Do not modify the "None".
    embedding_layer.build((None,))

    # Set the weights of the embedding layer to the embedding matrix. Your layer is now pretrained.
    embedding_layer.set_weights([emb_matrix])

    return embedding_layer    

"""
embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])
"""

def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    Function creating the Emojify-v2 model's graph.

    Arguments:
    input_shape -- shape of the input, usually (max_len,)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    model -- a model instance in Keras
    """
    sentence_indices = Input(shape = input_shape, dtype = 'int32')
    
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    
    embeddings = embedding_layer(sentence_indices)
    
    # Propagate the embeddings through an LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a batch of sequences.
    X = LSTM(128, return_sequences=True)(embeddings)#LSTM() has a flag called return_sequences to decide if you would like to return every hidden states or only the last one.
    X = Dropout(0.5)(X)
    
    # Propagate X trough another LSTM layer with 128-dimensional hidden state
    # Be careful, the returned output should be a single hidden state, not a batch of sequences.
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    
    X = Dense(5,activation='softmax')(X)
    
    # Add a softmax activation
    X = Activation('softmax')(X)
    
    model = Model(inputs=sentence_indices, outputs=X)
    
    return model
    
"""
maxLen = 10
model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary()
"""

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    
X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)
Y_train_oh = convert_to_one_hot(Y_train, C = 5)    
    
model.fit(X_train_indices, Y_train_oh, epochs = 50, batch_size = 32, shuffle=True)
    
    
X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)
print()
print("Test accuracy = ", acc) 
    
    
# This code allows you to see the mislabelled examples
C = 5
y_test_oh = np.eye(C)[Y_test.reshape(-1)]
X_test_indices = sentences_to_indices(X_test, word_to_index, maxLen)
pred = model.predict(X_test_indices)
for i in range(len(X_test)):
    x = X_test_indices
    num = np.argmax(pred[i])
    if(num != Y_test[i]):
        print('Expected emoji:'+ label_to_emoji(Y_test[i]) + ' prediction: '+ X_test[i] + label_to_emoji(num).strip())

#%%try it on your own example. Write your own sentence below.
# Change the sentence below to see your prediction. Make sure all the words are in the Glove embeddings.  
x_test = np.array(['zheng yu shi ge chu sheng '])
X_test_indices = sentences_to_indices(x_test, word_to_index, maxLen)
print(x_test[0] +' '+  label_to_emoji(np.argmax(model.predict(X_test_indices))))


#%%conclusion
"""
What you should remember: 
- If you have an NLP task where the training set is small, using word embeddings can help your algorithm significantly. Word embeddings allow your model to work on words in the test set that may not even have appeared in your training set. 
- Training sequence models in Keras (and in most other deep learning frameworks) requires a few important details: 
- To use mini-batches, the sequences need to be padded so that all the examples in a mini-batch have the same length. 
- An Embedding() layer can be initialized with pretrained values. These values can be either fixed or trained further on your dataset. If however your labeled dataset is small, it’s usually not worth trying to train a large pre-trained set of embeddings. 
- LSTM() has a flag called return_sequences to decide if you would like to return every hidden states or only the last one. 
- You can use Dropout() right after LSTM() to regularize your network.
"""










