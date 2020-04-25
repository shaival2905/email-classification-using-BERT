#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:45:42 2020

@author: nirav
"""

from gensim import corpora, models
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

df = pd.read_csv("preprocessed_Data_full_w_subject.csv")
data = [t.split() for t in df['subject'] if str(t)!="nan"]

word_freq = {}

#Find High Frequency Keywords And Remove Them
threshold = int(len(data)*0.2)
hig_freq_words = []

for i in data:
    for j in set(i):
        if j in word_freq:
            word_freq[j] += 1
        else:
            word_freq[j] = 1
        
        if word_freq[j] >= threshold:
            hig_freq_words.append(j)


hig_freq_words = set(hig_freq_words)
new_data = []

for i in data:
    d = []
    for j in i:
        if j not in hig_freq_words and len(j) > 2:
            d.append(j)
    new_data.append(d)
    
#Generate Dicitionary From The Data
dictionary = corpora.Dictionary(new_data)
#Generate Bag Of Words Represented Corpus
corpus = [dictionary.doc2bow(text) for text in new_data]

#Lda Model Training
ldamodel = models.ldamodel.LdaModel(corpus, num_topics=15, id2word = dictionary, passes=20)
#Find Topic Keywords for Each Topic
topic_keywords = ldamodel.print_topics(num_topics=15, num_words=3)

#Find Out Clustered Emails Based On Topic Keywords
for topic_keyword in topic_keywords:
    t1 = topic_keyword[1].split(" + ")[0].split("*")[-1].strip('"')
    t2 = topic_keyword[1].split(" + ")[1].split("*")[-1].strip('"')
    t3 = topic_keyword[1].split(" + ")[2].split("*")[-1].strip('"')
    
    for i in range(len(df)):
        if str(df.subject[i]) == "nan":
            continue
        if t1 in df.subject[i].split() :
            print(df.loc[i,])
            print(i)
    break
    print(topic_keyword)