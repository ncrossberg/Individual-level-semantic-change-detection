#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 29 09:33:18 2022

@author: nicolarossberg

Data Cleansing
"""

from nltk.corpus import stopwords
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()

#Importing Dataset
df = pd.read_csv('stormfront_data_full.csv')

#Removing non-enlgish posts
df = df.loc[df['stormfront_lang_id'] == 19]

#Removing posts with missing users
df = df.loc[df['stormfront_user'] != '[]']

#Making all writing lowercase
df = df.loc[df['stormfront_self_content'].str.lower()]

#Removing stopwords and non-alpha words
stop_words = stopwords.words('english')
df['stormfront_self_content'] = df['stormfront_self_content'].apply(lambda x: ' '.join([word for word in x.split() if word not in (stop_words) and word.isalpha()]))

#Creating list of post lengths
lenlist = []
for sentence in df['stormfront_self_content']:
    sentence = str(sentence)
    toksent = word_tokenize(sentence, 'english')
    lenlist.append(len(toksent))
    
#Creating a new column with the respective length of the user's post
df['Content_Length'] = lenlist

#Removing posts shorter than 15 words
df = df.loc[df['Content_Length'] >= 15]

#Removing excess columns
df = df.drop(columns=[ 'Unnamed: 0'])

#Removing users with less than 100 posts
# Create dictionary with number of posts per user
user_dict = {}
for user in df['stormfront_user']:
    if user in user_dict.keys():
        user_dict[user] += 1
    else:
        user_dict[user] = 1

#Create list with users whp have more than 100 posts
main_users = []
for entry in user_dict:
    if (user_dict[entry]) >= 100:
        main_users.append(entry)
        
users_to_remove = [x for x in user_dict if x not in main_users]

#Removing users
df = df[~df.stormfront_user.isin(users_to_remove)]

#Creating time bins
df['stormfront_publication_date'] = pd.to_datetime(df['stormfront_publication_date'])

df.sort_values(by = 'stormfront_docid')

for entry in main_users:
    tdf = df.loc[df['stormfront_user'] == entry]
    mini = min(tdf['stormfront_publication_date'])
    maxi = max(tdf['stormfront_publication_date'])
    bin_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    df.loc[df['stormfront_user'] == entry, 'time_bin'] = pd.qcut(tdf['stormfront_publication_date'], q = 10, labels = bin_labels)

bin_labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
name_list = []
post_list = []
bin_list = []
data = {'user' : name_list, 'posts' : post_list, 'time_bins' : bin_list}


post_dict = {}

for a, b, c in zip(df.stormfront_user, df.time_bin, df.stormfront_self_content):
    if (a, b) not in post_dict.keys():
        post_dict[(a, b)] = []
        post_dict[(a, b)].append(c)
    else:
        post_dict[(a, b)].append(c)

for key, value in post_dict.items():
    name_list.append(key[0])
    bin_list.append(key[1])
    post_list.append(value)


#Concatenating list of posts:
final_post = []
for lis in post_list:
    s = ','.join(lis)
    final_post.append(s)
    
df2 = pd.DataFrame()
df2['user'] = name_list
df2['posts'] = final_post
df2['time_bins'] = bin_list

#Removing punctuation
new_list = []
for post in df2['posts']:
    post = post.replace(',', '')
    post = post.replace("'", '')
    post = post.replace('[', '')
    post = post.replace(']', '')
    new_list.append(post)
    

df2['posts'] = new_list

#tokenizing posts
tok_list = []
for post in df2['posts']:
    post = word_tokenize(post, language='english')
    tok_list.append(post)
 
conc_list=[]
for lis in tok_list:
    s = ','.join(lis)
    conc_list.append(s)
    
final_list = []
for post in conc_list:
    post = post.lower()
    final_list.append(post.replace(",", " "))
    
    
  
df2['posts'] = final_list

for post in df2['posts']:
    post = post.replace(',', '')
    post = post.replace("'", '')
    post = post.replace('[', '')
    post = post.replace(']', '')
    new_list.append(post)
    
#Stemming
stem = []
sp = []

def convert(lst):
    return ' '.join(lst).split()
     



for post in df2['posts']:
    post = convert([post])
    stem = []
    for word in post:
        word = ps.stem(word)
        stem.append(word)
    sp.append(stem)
    
df2['stemmed_posts'] = sp
    
#Removing 1-letter words
postl = []

for post in df2['posts']:
    wordl = []
    for word in post:
        if len(word) > 1:
            wordl.append(word)
    postl.append(wordl)
  
  
df2['posts'] = postl

spostl = []

for post in df2['stemmed_posts']:
    swordl = []
    for word in post:
        if len(word) > 1:
            swordl.append(word)
    spostl.append(swordl)
    
df2['stemmed_posts'] = spostl

#Retaining only time bins 1 and 10
df3 = df2[df2['time_bins'].isin([1, 10])]

df3.to_csv('Cleansed_Data.csv')
