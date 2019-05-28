#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-23 18:36:41
@author: wind
'''



import jieba
import re
import random
from gensim.models import word2vec

def segment(src_file,out_file):
    with open(src_file,'r',encoding='utf-8') as sf,open(out_file,'w',encoding='utf-8') as of:
        for line in sf:
            label,sentences = line.strip().split("\t")
            sen_seq = re.split("[？。！?!]+",sentences) # 为什么没有英文句号？因为有小数点数据
            sen_arr = []
            for sen in sen_seq:
                if len(sen)>0:
                    sentence = "|-w-|".join(jieba.cut(sen))
                    sen_arr.append(sentence)
            of.write(label+"\t"+"|-s-|".join(sen_arr)+"\n")

def w2v(src_file,vectors_file):

    class FileIterable():
        def __init__(self,src_file):
            self.src_file = src_file

        def __iter__(self):
            with open(src_file,'r',encoding='utf-8') as f:
                for line in f:
                    _,sentences = line.strip().split("\t")
                    sen_arr = sentences.split("|-s-|")
                    for sen in sen_arr:
                        word_arr = sen.split("|-w-|")
                        yield word_arr

    model = word2vec.Word2Vec(FileIterable(src_file), hs=1,min_count=1,window=3,size=100)
    model.wv.save_word2vec_format(vectors_file,binary=False)

def top_k_vec(src_file,stop_file,tag_file,k):
    with open(stop_file,'r',encoding='utf-8') as f:
        stop = {l.strip() for l in f.readlines()}
    
    with open(src_file,'r',encoding='utf-8') as f:
        _,dim = f.readline().strip().split(" ") # 第一行不用
        dim = int(dim)

        pad = "<PAD>"+" 0"*dim
        unk = "<UNK>"+ " " + " ".join(["%.8f"%(random.random()*2-1) for _ in range(dim)])
        top_k = [pad,unk]
        for line in f:
            arr = line.strip().split(" ")
            if arr[0] not in stop:
                top_k.append(line.strip())
                if len(top_k)>=k:
                    break

    with open(tag_file,'w',encoding='utf-8') as f:
        f.write("\n".join(top_k))


def label_list(src_file,out_file):
    r = []
    with open(src_file,'r',encoding='utf-8') as sf:
        for line in sf:
            label,_ = line.strip().split("\t")
            r.append(label)
    r = sorted(list(set(r)))

    with open(out_file,'w',encoding='utf-8') as of:
        of.write("\n".join(r))


def file2id(src_file,label_file,word_file,out_file):
    label_dict = {}
    word_dict = {}
    with open(label_file,'r',encoding='utf-8') as f:
        i=0
        for line in f:
            label_dict[line.strip()] = i
            i+=1
    with open(word_file,'r',encoding='utf-8') as f:
        i=0
        for line in f:
            word_dict[line.strip().split(" ")[0]] = i
            i+=1
    
    with open(src_file,'r',encoding='utf-8') as sf,open(out_file,'w',encoding='utf-8') as of:
        for line in sf:
            label,sentences = line.strip().split("\t")
            label_id = label_dict[label]
            sen_arr = sentences.split("|-s-|")
            sen_ids = []
            for sen in sen_arr:
                word_arr = sen.split("|-w-|")
                word_ids = []
                for word in word_arr:
                    if word in word_dict: # 相比于textcnn,这里做了修改，不再保留unk字符，以减少句子长度
                        w_id = word_dict[word]
                        word_ids.append(str(w_id))
                word_str = " ".join(word_ids)
                sen_ids.append(word_str)
            of.write(str(label_id) + "\t" + "|".join(sen_ids) + "\n")

def plot_sentences_distribution(src_file):
    import matplotlib.pyplot as plt
    sen_len = []
    word_len = []
    with open(src_file,'r',encoding='utf-8') as f:
        for line in f:
            _,sentences = line.strip().split("\t")
            sen_arr = sentences.split("|")
            sen_len.append(len(sen_arr))
            for sen in sen_arr:
                word_len = len(sen.split(" "))
    plt.subplot(1,2,1)
    plt.hist(sen_len,bins=50)
    plt.subplot(1,2,2)
    plt.hist(word_len,bins=50)
    plt.show()
            
if __name__ == "__main__":
    # 对训练集和测试集进行分词
    segment("./data/cnews.train.txt","./data/cnews_seg_train.txt")
    segment("./data/cnews.test.txt","./data/cnews_seg_test.txt")
    # 用word2vec预训练词向量
    w2v("./data/cnews_seg_train.txt","./data/vectors")
    # 取出最常用的 20000 个词向量
    top_k_vec("./data/vectors","./data/stop_words.txt","./data/final_vectors",20000)
    # 创建标签文件
    label_list("./data/cnews_seg_train.txt","./data/label")
    # 将训练集测试集转化为id文件
    # textcnn中用了unk代表不常用的词，但句子太长了，有相当一部分句子长度在4000左右，
    # 甚至有极少数达到3-4w,这里把不常用词全从句子中删除，以减少计算量
    file2id("./data/cnews_seg_train.txt","./data/label","./data/final_vectors","./data/cnews_final_train.txt")
    file2id("./data/cnews_seg_test.txt","./data/label","./data/final_vectors","./data/cnews_final_test.txt")
    # 查看句子长度分布
    plot_sentences_distribution("./data/cnews_final_train.txt")