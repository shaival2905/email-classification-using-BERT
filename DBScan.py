#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:45:58 2020

@author: nirav
"""

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

#Find Top Keywords 
def find_keywords(df):
    frequencys = {}
    for sen in df['content']:
        for word in sen.split():
            if word in frequencys:
                frequencys[word]+=1
            else:
                frequencys[word]=1
    
    inv_dict={}
    for k,n in frequencys.items():
        if n in inv_dict:
            inv_dict[n].append(k)
        else:
            inv_dict[n]=[k]
    
    top_keywords_k = sorted(inv_dict.keys())[:1]
    top_keywords = []
    
    for key in top_keywords_k:
        top_keywords.extend(inv_dict[key])
        
    return top_keywords
    

df = pd.read_csv("preprocessed_Data.csv")

#Generate tagged data for doc2vecc
tagged_data = [TaggedDocument(words=word_tokenize(df['content'][i]), tags=[str(df['sender'][i])]) for i in range(len(df))]

#####################-Generate Doc2Vec Mode-##################################
max_epochs = 100
vec_size = 20
alpha = 0.025

model = Doc2Vec(size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)
  
model.build_vocab(tagged_data)

for epoch in range(max_epochs):
    print('iteration {0}'.format(epoch))
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.iter)
    # decrease the learning rate
    model.alpha -= 0.0002
    # fix the learning rate, no decay
    model.min_alpha = model.alpha

model.save("d2v.model")
###############################################################################

print("Model Saved")

model= Doc2Vec.load("d2v.model")

#Convert all emails into doc2vec representation
data = [model.infer_vector(word_tokenize(df['content'][i])) for i in range(len(df))]   


#Bert_Vectors = []
#X_file_init = 'train_encodings_batch_'
#train_data_path="Train_data"
#
#for i in range(100):
#    j=181
#    batch_X_file = train_data_path + '/' + X_file_init + '{}.npy'.format(j)
#    X_batch = np.load(batch_X_file, allow_pickle=True)
#    Bert_Vectors.extend(X_batch)
# Data = Bert_Vectors

#Run it through DBSCAN clustering algorithm
clustering = DBSCAN(eps=0.11, min_samples=3, metric='cosine').fit(data)               
clusters = clustering.labels_
df['cluster'] = clusters
groups = df.groupby('cluster')

for group, df_group in groups:
    if group == -1 :
        continue
    print("Group No: ", group+1, " | ", "No of Emails: ",len(df_group))
    
###Select eps
neigh = NearestNeighbors(n_neighbors=2,metric='cosine')
nbrs = neigh.fit(data)
distances, indices = nbrs.kneighbors(data)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
plt.grid(True)
plt.xlabel("Frequency")
plt.ylabel("Distance")
plt.plot(distances)
    
