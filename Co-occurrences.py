#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:47:32 2022

@author: nicolarossberg
"""

import pandas as pd
import itertools
from collections import defaultdict
import numpy as np
from tqdm import tqdm

#Defining function
def co_occurrence(sentences, window_size):
    d = defaultdict(int)
    vocab = set()
    for text in sentences:
        # preprocessing (use tokenizer instead)
        text = text.lower().split()
        # iterate over sentences
        for i in range(len(text)):
            token = text[i]
            vocab.add(token)  # add to vocab
            next_token = text[i+1 : i+1+window_size]
            for t in next_token:
                key = tuple( sorted([t, token]) )
                d[key] += 1
    
    vocab = sorted(vocab) # sort vocab
    df = pd.DataFrame(data=np.zeros((len(vocab), len(vocab)), dtype=np.int16),
                      index=vocab,
                      columns=vocab)
    for key, value in d.items():
        df.at[key[0], key[1]] = value
        df.at[key[1], key[0]] = value
    # df.to_excel('co-oc.xlsx')
    return df

#Reading in Data
df = pd.read_csv('25.3.csv')

#Converting data
df['stemmed posts'] = df['stemmed posts'].str.replace(',','')
df['stemmed posts'] = df['stemmed posts'].str.replace("'",'')
df['stemmed posts'] = df['stemmed posts'].str.replace('[','')
df['stemmed posts'] = df['stemmed posts'].str.replace(']','')

#Dropping unnecessary columns 
df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])


df["key"] = df["user"] + df["time_bins"].astype('string')
df["cooccurrence dict"] = ""

save = False #<- set to true to save for each user
for index,row in tqdm(df.iterrows()):
    # get key ids
    username = row['user']
    time_bin = row['time_bins']
    filename = "CoOccurrence_" + username + "_" + str(time_bin) + ".csv"
    
    post = row['stemmed posts']
    smalldic = {}
    sen = [post, post]
    co_oc_mat = co_occurrence(sen, 5)
    d = pd.DataFrame(co_oc_mat)
    name_list = d.columns.tolist()

    perm_iterator = itertools.permutations(name_list, 2)
    Output = set(tuple(sorted(t)) for t in perm_iterator)
    for i, j in Output:
        if d.at[i, j] > 0:
            smalldic[(i, j)] = d.at[i, j]
            
    df.at[index,"cooccurrence dict"] = str(smalldic)
   
    if save:
        codf = pd.DataFrame()
        codf["Word_Combo"] = smalldic.keys()
        codf["Co_Occurrences"] = smalldic.values()
        codf.to_csv(filename,chunksize=100,mode='a',index=False)


