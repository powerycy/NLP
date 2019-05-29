#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-24 18:36:07
@author: wind
'''
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self,data_file,max_word_len,max_sen_len):
        super().__init__()

        self.data = []
        
        with open(data_file,'r',encoding='utf-8') as f: 
            for line in tqdm(f,desc="加载数据"):
                arr = line.strip().split("\t")
                if len(arr)<2:
                    continue

                label_id,sentence = arr
                # label
                label_id = int(label_id)

                x = []
                lengths = []

                sent_arr = sentence.split("|")
                for sent in sent_arr:
                    if len(sent)<=0:
                        continue
                    word_arr = sent.split(" ")
                    word_ids = [int(w) for w in word_arr]
                    word_ids = word_ids[:max_word_len]
                    word_len = len(word_ids)
                    # 词级别 padding 
                    padding = [0]*(max_word_len-word_len)
                    word_ids += padding

                    x.append(word_ids)
                    lengths.append(word_len)
                
                # 句子级别 padding
                x = x[:max_sen_len]
                lengths = lengths[:max_sen_len]
                pad_len = max_sen_len - len(x)
                
                padding = [[0]*max_word_len]*pad_len
                x += padding
                lengths += [0]*pad_len

                self.data.append({
                    "label_id":torch.LongTensor([label_id]).to('cpu'),
                    "x":torch.LongTensor(x).to("cpu"),
                    "lengths":torch.LongTensor(lengths).to('cpu')
                })
    
    def __getitem__(self,item):
        return self.data[item]
    
    def __len__(self):
        return len(self.data)

def get_pre_embedding_matrix(src_file):
    arr = []
    with open(src_file,'r',encoding='utf-8') as f:
        for line in f:
            vec = line.strip().split(" ")[1:]
            vec = [float(i) for i in vec]
            arr.append(vec)
    return arr

def get_labels(src_file):
    with open(src_file,'r',encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]
        

if __name__ == "__main__":
    # ds = NewsDataset("./data/cnews_final_test.txt",300)
    # print(ds[0])
    # get_pre_embedding_matrix("./data/final_vectors")
    print(get_labels('./data/label'))
