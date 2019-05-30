#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-23 18:36:41
@author: wind
'''

import os
from pytorch_pretrained_bert import BertTokenizer

def label_list(src_file,out_file):
    r = []
    with open(src_file,'r',encoding='utf-8') as sf:
        for line in sf:
            label,_ = line.strip().split("\t")
            r.append(label)
    r = sorted(list(set(r)))

    with open(out_file,'w',encoding='utf-8') as of:
        of.write("\n".join(r))


def file2id(src_file,label_file,out_file):
    label_dict = {}
    with open(label_file,'r',encoding='utf-8') as f:
        i=0
        for line in f:
            label_dict[line.strip()] = i
            i+=1
    
    with open(src_file,'r',encoding='utf-8') as sf,open(out_file,'w',encoding='utf-8') as of:
        
        if os.path.isfile("./bert/bert-base-chinese-vocab.txt"):
            tokenizer = BertTokenizer.from_pretrained('./bert/bert-base-chinese-vocab.txt')
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        tokenizer.max_len = 1e12

        for line in sf:
            label,sentences = line.strip().split("\t")
            label_id = label_dict[label]

            tokenized_text = tokenizer.tokenize(sentences)
            text_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
            text_ids = [str(w) for w in text_ids]

            of.write(str(label_id) + "\t" + " ".join(text_ids) + "\n")

def plot_sentences_distribution(src_file):
    import matplotlib.pyplot as plt
    sent_len = []
    with open(src_file,'r',encoding='utf-8') as f:
        sent_len = [len(line.strip().split(" ")) for line in f.readlines()]
    plt.hist(sent_len,bins=100)
    plt.show()
            
if __name__ == "__main__":
    # 创建标签文件
    label_list("./data/cnews.train.txt","./data/label")
    # 将训练集测试集转化为id文件
    file2id("./data/cnews.train.txt","./data/label","./data/cnews_final_train.txt")
    file2id("./data/cnews.test.txt","./data/label","./data/cnews_final_test.txt")
    # 查看句子长度分布
    plot_sentences_distribution("./data/cnews_final_train.txt")