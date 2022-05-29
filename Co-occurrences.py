#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 11:47:32 2022

@author: nicolarossberg
"""

import pandas as pd
from nltk import bigrams
import itertools
from itertools import chain
import collections
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import time
import csv

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

df = pd.read_csv('25.3.csv')

df['stemmed posts'] = df['stemmed posts'].str.replace(',','')
df['stemmed posts'] = df['stemmed posts'].str.replace("'",'')
df['stemmed posts'] = df['stemmed posts'].str.replace('[','')
df['stemmed posts'] = df['stemmed posts'].str.replace(']','')
 
df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])



# small_df = df.iloc[2197:2198,1:]
# small_df["key"] = small_df["user"] + small_df["time_bins"].astype('string')
# small_df["cooccurrence dict"] = ""

df["key"] = df["user"] + df["time_bins"].astype('string')
df["cooccurrence dict"] = ""

save = True
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

# transform a cooccurrence cell into a much nicer to use dict
import ast

codict = ast.literal_eval(df["cooccurrence dict"][0])

    



"""
col_name= ['dic']

for row in tqdm(df['stemmed posts']):
    smalldic = {}
    smalllis = []
    sen = [post, post]
    co_oc_mat = co_occurrence(sen, 5)
    d = pd.DataFrame(co_oc_mat)
    name_list = d.columns.tolist()
    perm_iterator = itertools.permutations(name_list, 2)
    Output = set(tuple(sorted(t)) for t in perm_iterator)
    for i, j in Output:
        if d.at[i, j] > 2:
            smalldic[(i, j)] = d.at[i, j]
    smalllis.append(smalldic)
    fd =  open('practicefivegrams1.csv', 'a')
    csv_writer = csv.writer(fd)
    csv_writer.writerow([[item] for item in smalllis])
    fd.close()
    break
"""
    