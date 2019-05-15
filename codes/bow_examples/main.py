#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-15 13:21:40
@author: wind
'''

# 词袋法
from sklearn.feature_extraction.text import CountVectorizer  
vectorizer=CountVectorizer()
corpus=["I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"] 
print(vectorizer.fit_transform(corpus))

print(vectorizer.fit_transform(corpus).toarray())
print(vectorizer.get_feature_names())

# Hash Trick
from sklearn.feature_extraction.text import HashingVectorizer 
vectorizer2=HashingVectorizer(n_features = 6,norm = None)
print(vectorizer2.fit_transform(corpus))


# tf-idf 方法1

from sklearn.feature_extraction.text import TfidfTransformer  
from sklearn.feature_extraction.text import CountVectorizer  

corpus=["I come to China to travel", 
    "This is a car polupar in China",          
    "I love tea and Apple ",   
    "The work is to write some papers in science"] 

vectorizer=CountVectorizer()

transformer = TfidfTransformer()
tfidf = transformer.fit_transform(vectorizer.fit_transform(corpus))  
print(tfidf)

# tf-idf 方法2
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf2 = TfidfVectorizer()
re = tfidf2.fit_transform(corpus)
print(re)


