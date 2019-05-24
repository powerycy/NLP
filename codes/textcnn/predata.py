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
            label,sentence = line.strip().split("\t")
            sentence = " ".join(jieba.cut(sentence))
            sentence = re.sub("\s+"," ",sentence)
            of.write(label+"\t"+sentence+"\n")

def w2v(src_file,vectors_file):
    sentences = word2vec.LineSentence(src_file)
    model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100)
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
            label,segment = line.strip().split("\t")
            label_id = label_dict[label]
            word_ids = []
            for word in segment.split(" "):
                if word in word_dict:
                    w_id = word_dict[word]
                else:
                    w_id = word_dict["<UNK>"]
                word_ids.append(str(w_id))
            of.write(str(label_id) + "\t" + " ".join(word_ids) + "\n")
            
if __name__ == "__main__":
    # segment("./data/cnews.train.txt","./data/cnews_seg_train.txt")
    # segment("./data/cnews.test.txt","./data/cnews_seg_test.txt")
    # w2v("./data/cnews_seg_train.txt","./data/vectors")
    # top_k_vec("./data/vectors","./data/stop_words.txt","./data/final_vectors",5000)
    # label_list("./data/cnews_seg_train.txt","./data/label")
    # file2id("./data/cnews_seg_train.txt","./data/label","./data/final_vectors","./data/cnews_final_train.txt")
    # file2id("./data/cnews_seg_test.txt","./data/label","./data/final_vectors","./data/cnews_final_test.txt")