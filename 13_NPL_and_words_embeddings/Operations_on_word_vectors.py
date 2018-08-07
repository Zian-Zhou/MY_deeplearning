# -*- coding: utf-8 -*-
"""
Created on Mon Jul 30 13:35:39 2018

@author: a
"""

###################################Operations on word vectors

#Because word embeddings are very computionally expensive to train, 
#most ML practitioners will load a pre-trained set of embeddings.

import numpy as np
from w2v_utils import *

words, word_to_vec_map = read_glove_vecs('data/glove.6B.50d.txt')
"""
- words: set of words in the vocabulary. 
- word_to_vec_map: dictionary mapping words to their GloVe vector representation.
"""
#%%```````````````````````````````1 - Cosine similarity
#To measure how similar two words are, we need a way to measure the degree of similarity between two embedding vectors for the two words.
#CosineSimilarity(u, v)=(u.v) / (||u||2*||v||2)=cos(θ)
def cosine_similarity(u,v):
    """
    Cosine similarity reflects the degree of similariy between u and v

    Arguments:
        u -- a word vector of shape (n,)          
        v -- a word vector of shape (n,)

    Returns:
        cosine_similarity -- the cosine similarity between u and v defined by the formula above.
    """
#    distance = 0.0
    
    dot = np.dot(u,v)
    norm_u = np.sqrt(np.sum(u**2))
    norm_v = np.sqrt(np.sum(v**2))
    
    consine_similarity = dot / (norm_u*norm_v)
    
    return consine_similarity

"""
father = word_to_vec_map["father"]
mother = word_to_vec_map["mother"]
ball = word_to_vec_map["ball"]
crocodile = word_to_vec_map["crocodile"]
france = word_to_vec_map["france"]
italy = word_to_vec_map["italy"]
paris = word_to_vec_map["paris"]
rome = word_to_vec_map["rome"]

print("cosine_similarity(father, mother) = ", cosine_similarity(father, mother))
print("cosine_similarity(ball, crocodile) = ",cosine_similarity(ball, crocodile))
print("cosine_similarity(france - paris, rome - italy) = ",cosine_similarity(france - paris, rome - italy))
"""

#%%````````````````````````````````2 - Word analogy task
"""
n the word analogy task, we complete the sentence “a is to b as c is to __”. 
An example is ‘man is to woman as king is to queen’ . 
In detail, we are trying to find a word d, 
such that the associated word vectors ea,eb,ec,ed are related in the following manner: eb−ea≈ed−ec.
We will measure the similarity between eb−ea and ed−ec using cosine similarity.
"""
def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    Performs the word analogy task as explained above: a is to b as c is to ____. 

    Arguments:
    word_a -- a word, string
    word_b -- a word, string
    word_c -- a word, string
    word_to_vec_map -- dictionary that maps words to their corresponding vectors. 

    Returns:
    best_word --  the word such that v_b - v_a is close to v_best_word - v_c, as measured by cosine similarity
    """
    word_a, word_b, word_c = word_a.lower(),word_b.lower(),word_c.lower()
    
    e_a,e_b,e_c = word_to_vec_map[word_a],word_to_vec_map[word_b],word_to_vec_map[word_c]
    
    words = word_to_vec_map.keys()
    max_cosine_sim = -100
    best_word = None
    
    for w in words:
        if w in [word_a,word_b,word_c]:
            continue
        cosine_sim = cosine_similarity(e_b-e_a, word_to_vec_map[w] - e_c)
        
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = w
            
    return best_word
"""
triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
for triad in triads_to_try:
    print ('{} -> {} :: {} -> {}'.format( *triad, complete_analogy(*triad,word_to_vec_map)))
"""
    
"""
Cosine similarity a good way to compare similarity between pairs of word vectors. (Though L2 distance works too.)
For NLP applications, using a pre-trained set of word vectors from the internet is often a good way to get started.
"""

#%%```````````````````````````3 - Debiasing word vectors (OPTIONAL/UNGRADED)
#性别倾向、种族倾向

#3.1 - Neutralize bias for non-gender specific words
"""
e^bias_component=(e⋅g)/(||g||2^2)∗g   ---e在g方向上的投影
e^debiased=e−e^bias_component
"""
def neutralize(word,g,word_to_vec_map):
    """
    Removes the bias of "word" by projecting it on the space orthogonal to the bias axis. 
    This function ensures that gender neutral words are zero in the gender subspace.

    Arguments:
        word -- string indicating the word to debias
        g -- numpy-array of shape (50,), corresponding to the bias axis (such as gender)
        word_to_vec_map -- dictionary mapping words to their corresponding vectors.

    Returns:
        e_debiased -- neutralized word vector representation of the input "word"
    """
    e = word_to_vec_map[word]
    e_biascomponent = np.dot(e,g) / np.square(np.linalg.norm(g)) * g
    e_debiased = e - e_biascomponent
    
    return e_debiased
"""
g = word_to_vec_map['woman'] - word_to_vec_map['man']
#print(g)

e = "receptionist"
print("cosine similarity between " + e + " and g, before neutralizing: ", cosine_similarity(word_to_vec_map["receptionist"], g))

e_debiased = neutralize("receptionist", g, word_to_vec_map)
print("cosine similarity between " + e + " and g, after neutralizing: ", cosine_similarity(e_debiased, g))
"""

#3.2 - Equalization algorithm for gender-specific words

"""#
μ=(e_w1+e_w2) /2-------------------------------------------------------------(4)

μB=μ⋅bias_axis / ||bias_axis||2^2 ∗ bias_axis--------------------------------(5)

μ⊥=μ−μB---------------------------------------------------------------------(6)

e_w1B=ew1⋅bias_axis / ||bias_axis||2^2 ∗ bias_axis---------------------------(7)
e_w2B=ew2⋅bias_axis / ||bias_axis||2^2 ∗ bias_axis---------------------------(8)

e^corrected_w1B=sqrt(| 1− ||μ⊥||2^2 |) ∗ (e_w1B−μB) / |(e_w1−μ⊥)−μB)|------(9)
e^corrected_w2B=sqrt(| 1− ||μ⊥||2^2 |) ∗ (e_w2B−μB) / |(e_w2−μ⊥)−μB)|-----(10)
    
e1=e^corrected_w1B + μ⊥----------------------------------------------------(11)
e2=e^corrected_w2B + μ⊥----------------------------------------------------(12)
"""

def equalize(pair, bias_axis, word_to_vec_map):
    """
    Debias gender specific words by following the equalize method described in the figure above.

    Arguments:
    pair -- pair of strings of gender specific words to debias, e.g. ("actress", "actor") 
    bias_axis -- numpy-array of shape (50,), vector corresponding to the bias axis, e.g. gender
    word_to_vec_map -- dictionary mapping words to their corresponding vectors

    Returns
    e_1 -- word vector corresponding to the first word
    e_2 -- word vector corresponding to the second word
    """
    w1, w2 = pair
    e_w1,e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]
    
    mu = (e_w1 + e_w2) / 2
    
    mu_B = np.dot(mu,bias_axis) / np.sum(bias_axis**2) * bias_axis
    mu_orth = mu - mu_B
    
    e_w1B = np.dot(e_w1,bias_axis) / np.sum(bias_axis**2) * bias_axis
    e_w2B = np.dot(e_w2,bias_axis) / np.sum(bias_axis**2) * bias_axis
    
    corrected_e_w1B = np.sqrt(np.abs(1-np.sum(mu_orth**2))) * (e_w1B - mu_B) / np.linalg.norm(e_w1 - mu_orth - mu_B)
    corrected_e_w2B = np.sqrt(np.abs(1-np.sum(mu_orth**2))) * (e_w2B - mu_B) / np.linalg.norm(e_w2 - mu_orth - mu_B)
    
    e1 = corrected_e_w1B + mu_orth
    e2 = corrected_e_w2B + mu_orth
    
    return e1,e2

"""
print("cosine similarities before equalizing:")
print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
print()
e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
print("cosine similarities after equalizing:")
print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
"""
    




































