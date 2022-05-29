#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 12:43:12 2022

@author: nicolarossberg
"""

import pandas as pd
import ast
from tqdm import tqdm
from math import sqrt
import gensim
from gensim.models import KeyedVectors
from gensim import downloader

wv = KeyedVectors.load("word2vec.wordvectors", mmap='r')
glove_vectors = gensim.downloader.load('glove-twitter-200')


df = pd.read_csv('25.3.csv')

user_set = set()
for row in df['user']:
    user_set.add(row)
    
new_list = list(user_set)

#Defining relevant functions
def square_rooted(x):

    return round(sqrt(sum([a*a for a in x])), 3)


def cosine_similarity(x, y):

    input1 = {}
    input2 = {}
    vector2 = []
    vector1 = []

    if len(x) > len(y):
        input1 = x
        input2 = y
    else:
        input1 = y
        input2 = x

    vector1 = list(input1.values())

    for k in input1.keys():    # Normalizing input vectors.
        if k in input2:
            # picking the values for the common keys from input 2
            vector2.append(float(input2[k]))
        else:
            vector2.append(float(0))

    try:
        numerator = sum(a*b for a, b in zip(vector2, vector1))
        denominator = square_rooted(vector1)*square_rooted(vector2)
        return round(numerator/float(denominator), 3)
    except:
        return 0


def closest_neighbor(w):
    """finds the three closest neighbors for the word in question"""

    cnt = []
    cnt.append(glove_vectors.most_similar(w)[0:3])
    cn = []
    cn = [[tup[0] for tup in word] for word in cnt][0]
    return cn


print(closest_neighbor('hello'))


save = False #<- Set to true to save output
for user in tqdm(new_list):
    # Create dictionaries for first and 10th time bin
    si1dt = {}
    si10dt = {} 
    si1dict = {}
    si10dict = {}
    bothdict = {}
    username = str(user)
    fn = "Embed_" + username + ".csv"
    filename = "Sim_" + username + ".csv"
    filename1 = "CoOccurrence_" + username + "_" + str(1) + ".csv"
    filename10 = "CoOccurrence_" + username + "_" + str(10) + ".csv"
    tdf1 = pd.read_csv(str(filename1))
    tdf10 = pd.read_csv(str(filename10))
    tdf1.Word_Combo = tdf1.Word_Combo.apply(ast.literal_eval)
    tdf10.Word_Combo = tdf10.Word_Combo.apply(ast.literal_eval)
    di1 = dict(zip(tdf1.Word_Combo, tdf1.Co_Occurrences))
    di10 = dict(zip(tdf10.Word_Combo, tdf10.Co_Occurrences))
    # Create list of words that are in both time bins/ identifying potential words for change
    si1 = set([a for b in di1.keys() for a in b])
    si10 = set([a for b in di10.keys() for a in b])
    siboth = set([w for w in si1 if w in si10])
    siboth.add(w for w in si10 if w in si1)
    # Add word if nearest neighbor in other list/ i.e. create embedding-expanded dictionary
    si1tmp = set([w for w in si1 if w not in si10])
    si10tmp = set([w for w in si10 if w not in si1])
    for word in si1tmp:
        try:
            cn = closest_neighbor(word)
            si1dt[word] = cn
        except:
            continue
    for word in si10tmp:
        try:
            cn = closest_neighbor(word)
            si10dt[word] = cn
        except:
            continue       
    for key, value in si1dt.items():
        for i in value:
            if i in si10:
                si1dict[key] = i
    for key, value in si10dt.items():
        for i in value:
            if i in si1:
                si10dict[key] = i
    for word in si1dict.keys():
        #words in si1 with closest neighbor in si10
        dict1 = {}
        dict10 = {}
        neiword = si1dict[word]
        # dictionary with word in si1 and nearest neighbore of word in si10
        for key, value in di1.items():
            if word in key[0]:
                dict1[key[1]] = value
            elif word in key[1]:
                dict1[key[0]] = value
        for key, value in di10.items():
            if neiword in key[0]:
                dict10[key[1]] = value
            elif neiword in key[1]:
                dict10[key[0]] = value
        cos = cosine_similarity(dict1, dict10)
        bothdict[word] = cos
    for word in si10dict.keys():
        neiword = si10dict[word]
        # dictionary with word in si1 and nearest neighbore of word in si10
        for key, value in di10.items():
            if word in key[0]:
                dict10[key[1]] = value
            elif word in key[1]:
                dict10[key[0]] = value
        for key, value in di1.items():
            if neiword in key[0]:
                dict1[key[1]] = value
            elif neiword in key[1]:
                dict1[key[0]] = value
        cos = cosine_similarity(dict1, dict10)
        bothdict[word] = cos
    
    if save:
        CoSim = pd.DataFrame()
        CoSim["Embed-Word"] = bothdict.keys()
        CoSim["Similarity"] = bothdict.values()
        CoSim.to_csv(fn, chunksize=100, mode='a', index=False)


