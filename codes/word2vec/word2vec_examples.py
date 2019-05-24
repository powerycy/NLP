#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-19 17:26:29
@author: wind
'''

import jieba
from gensim.models import word2vec

# 分词
jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('田国富', True)
jieba.suggest_freq('高育良', True)
jieba.suggest_freq('侯亮平', True)
jieba.suggest_freq('钟小艾', True)
jieba.suggest_freq('陈岩石', True)
jieba.suggest_freq('欧阳菁', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('蔡成功', True)
jieba.suggest_freq('孙连城', True)
jieba.suggest_freq('季昌明', True)
jieba.suggest_freq('丁义珍', True)
jieba.suggest_freq('郑西坡', True)
jieba.suggest_freq('赵东来', True)
jieba.suggest_freq('高小琴', True)
jieba.suggest_freq('赵瑞龙', True)
jieba.suggest_freq('林华华', True)
jieba.suggest_freq('陆亦可', True)
jieba.suggest_freq('刘新建', True)
jieba.suggest_freq('刘庆祝', True)

with open('./in_the_name_of_people.txt','r',encoding='utf-8') as rf,\
    open('./in_the_name_of_people_segment.txt','w',encoding='utf-8') as wf:
    
    document = rf.read()
    document_cut = jieba.cut(document)
    result = " ".join(document_cut)

    wf.write(result)

# 训练模型
sentences = word2vec.LineSentence('./in_the_name_of_people_segment.txt')
model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100)

# 查看模型效果
print("-"*10,"5个与'李达康'相近的3个字的词：","-"*10)
req_count = 5
for key in model.wv.similar_by_word("李达康",topn=100):
    if len(key[0])==3:
        req_count -=1
        print(key)
        if req_count == 0:
            break

print("-"*10,"5个与'陆亦可'相近的3个字的词：","-"*10)
req_count = 5
for key in model.wv.similar_by_word("陆亦可",topn=100):
    if len(key[0])==3:
        req_count -=1
        print(key)
        if req_count == 0:
            break
            

print("-"*10,"两个词之间的相似度","-"*10)
print("沙瑞金 高育良：",model.wv.similarity("沙瑞金","高育良"))
print("高育良 沙瑞金：",model.wv.similarity("高育良","沙瑞金"))
print("李达康 侯亮平：",model.wv.similarity("李达康","侯亮平"))


print("-"*10,"给定列表中的哪个单词与其他单词不一致","-"*10)
print("沙瑞金 高育良 李达康 侯亮平：",model.wv.doesnt_match("沙瑞金 高育良 李达康 侯亮平".split()))
print("沙瑞金 高育良 李达康 刘庆祝：",model.wv.doesnt_match("沙瑞金 高育良 李达康 刘庆祝".split()))

# 保存模型，保存词向量,加载模型
# model.save("./word2vec_gensim")
# model.wv.save_word2vec_format("./vectors", # 词与向量，
#                                   "./vocabulary", # 词与词频
#                                   binary=False)
# model.wv.save("./kv") # 二进制文件