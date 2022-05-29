#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 24 12:07:36 2022

@author: nicolarossberg
"""
#Before you run this make sure that only the non-word part of the tuple is in the dictionary; otherwise tuples in the wrong order will decrease similarity
import pandas as pd
import ast
from tqdm import tqdm
from math import sqrt

# Replacing problematic Names
df = pd.read_csv('Cleansed_Data.csv')
#Chaning column names
df = df.rename(columns={"Username":"user"})
df = df.rename(columns={"Time bin":"time_bins"})
#Changing problematic user names
#Checking indeces
df.index[df['user'] == 'Commander/NSM']
df.index[df['user'] == 'c/c++ coder']
#Fixing names
df.at[3088, 'user'] = 'CommanderNSM'
df.at[3092, 'user'] = 'CommanderNSM'
df.at[2075, 'user'] = 'cc++ coder'
df.at[2115, 'user'] = 'cc++ coder'

#Concerting to new Dataset
df.to_csv('25.3.git.csv')

#Defining necessary functions
def square_rooted(x):

    return round(sqrt(sum([a*a for a in x])),3)

def cosine_similarity(x,y):

    input1 = {}
    input2 = {}
    vector2 = []
    vector1 =[]

    if len(x) > len(y):
        input1 = x
        input2 = y
    else:
        input1 = y
        input2 = x


    vector1 = list(input1.values())

    for k in input1.keys():    # Normalizing input vectors. 
        if k in input2:
            vector2.append(float(input2[k])) #picking the values for the common keys from input 2
        else :
            vector2.append(float(0))
    
    try:
        numerator = sum(a*b for a,b in zip(vector2,vector1))
        denominator = square_rooted(vector1)*square_rooted(vector2)
        return round(numerator/float(denominator),3)
    except:
        return 0

#Reading in Data
df = pd.read_csv('25.3.git.csv')

#Creating list of users
user_set = set()

for row in df['user']:
    user_set.add(row)
    
#Concerting set to list
user_list = list(user_set)


#Creating co-occurrence Matrix
save = False #<- set to true to save each user's co-occurrences
for item in tqdm(user_set):
    #Create dictionaries for first and 10th time bin
    big_dict = {}
    username = str(item)
    filename = "Sim_" + username + ".csv"
    filename1 = "CoOccurrence/CoOccurrence_" + username + "_" + str(1) + ".csv"
    filename10 = "CoOccurrence/CoOccurrence_" + username + "_" + str(10) + ".csv"
    tdf1 = pd.read_csv(str(filename1))
    tdf10 = pd.read_csv(str(filename10))
    tdf1.Word_Combo = tdf1.Word_Combo.apply(ast.literal_eval)
    tdf10.Word_Combo = tdf10.Word_Combo.apply(ast.literal_eval)
    di1 = dict(zip(tdf1.Word_Combo, tdf1.Co_Occurrences))
    di10 = dict(zip(tdf10.Word_Combo, tdf10.Co_Occurrences))
    #Create list of words that are in both time bins/ identifying potential words for change
    si1 = set()
    si10 = set()
    for a, b in di1.keys():
        si1.add(a)
        si1.add(b)
    for c, d in di10.keys():
        si10.add(c)
        si10.add(d)
    siboth = set([x for x in si1 if x in si10])
    #Calculate associations each word in each time bin
    for word in siboth:
        print(word)
        dict1 = {}
        dict10 = {}
        for key, value in di1.items():
            if word in key:
                if word in key[0]:
                    dict1[key[1]] = value
                elif word in key[1]:
                    dict1[key[0]] = value
        for key, value in di10.items():
            if word in key:
                if word in key[0]:
                    dict10[key[1]] = value
                elif word in key[1]:
                    dict10[key[0]] = value
        cos = float(cosine_similarity(dict1, dict10))
        big_dict[word] = cos

    if save:
        CoSim = pd.DataFrame()
        CoSim["Word"] = big_dict.keys()
        CoSim["Similarity"] = big_dict.values()
        CoSim.to_csv(filename,chunksize=100,mode='a',index=False)

