#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 00:05:16 2020

@author: nirav
"""

import nltk,re
import os
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer 
import pandas as pd

def stem_filter(txt, stem=False):
    stop_list = stopwords.words('english')
    stemmer = SnowballStemmer('english')
    txt = re.sub("#\S+|&\S+|@\S+|https?:\S+|RT|[^A-Za-z0-9]+",' ', str(txt).lower()).strip()
    txt = re.sub("&\S*|@\S+|https?:\S+",' ', str(txt).lower()).strip()
    txt = re.sub("[^A-Za-z']+",' ',str(txt).lower()).strip()
    tokens = []
    for token in txt.split():
        if token not in stop_list:
            if stem:
                tokens.append(stemmer.stem(token))
            else:
                tokens.append(token)
    return " ".join(tokens)


def read_and_filter_data(folder):

    data   = pd.DataFrame(columns=['sender','content'])
    
    people_list=['mann-k',
    'kaminski-v',
    'dasovich-j',
    'germany-c',
    'shackleton-s',
    'jones-t',
    'bass-e',
    'lenhart-m',
    'beck-s',
    'symes-k',
    'scott-s',
    'taylor-m',
    'love-p',
    'arnold-j',
    'perlingiere-d']
    
    for people in os.listdir(folder):
        
        if people not in people_list:
            continue
        
        input_fold = "_sent_mail"
        fname = people.split("-")[0]
        input_file_name = os.path.join(os.path.join(folder,people),input_fold)
        
        if os.path.exists(input_file_name): 
            for file in os.listdir(input_file_name):
                inp_file = os.path.join(input_file_name,file)
                txt_file = open(inp_file)
                content = txt_file.read()
                
                paragraphs = content.split("\n\n")
                from_people = True
                filtered = []
                
                for paragraph in paragraphs:
                    #print("#####")
                    #print(paragraph)
                    skip = False
                    for sentences in paragraph.split("\n"):
                        if "to" in sentences.lower() and fname in sentences.lower() and "x-to" not in sentences.lower():
                            from_people=False
                            break
                        if "cc" in sentences.lower() and fname in sentences.lower():
                            from_people=False
                            break
                        if "@enron.com" in sentences.lower():
                            skip=True
                            break
                        
                        if "forwarded" in sentences.lower():
                            skip=True
                            break
                    
                    if skip:
                        continue
                    if not(from_people):
                        break
                    
                filtered.append(paragraph)
                    
                stemmed_data = stem_filter("\n".join(filtered))
                    
                if stemmed_data != "":
                    data.loc[len(data)]=[people,stemmed_data]
                        
    
    data.to_csv("Kamno_data.csv",index=False)
    
    return data

input_dir = "maildir"
data = read_and_filter_data(input_dir)
    