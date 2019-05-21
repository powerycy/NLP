#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-21 15:23:11
@author: wind
'''

from gensim.models import FastText

# 训练
sentences = [["你", "是", "谁"], ["我", "是", "中国人"]]
# 方法一（官方不建议这样用）
# model = FastText(sentences,  size=4, window=3, min_count=1, iter=10,min_n = 3 , max_n = 6,word_ngrams = 1)
# 方法二
model = FastText(size=4, window=3, min_count=1, word_ngrams=1)
model.build_vocab(sentences=sentences)
model.train(sentences=sentences, total_examples=len(sentences), epochs=10)  

# 获取词向量
print(model.wv['你']) # 词向量获得的方式
print(model.wv.word_vec('你'))

# 保存模型
model.save('./model.bin')
# 加载模型
model = FastText.load("./model.bin")

# 保存词向量
model.wv.save_word2vec_format("./wv.txt")
