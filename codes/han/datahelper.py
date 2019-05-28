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
    def __init__(self,data_file,max_seq_len):
        super().__init__()

        self.data = []
        
        with open(data_file,'r',encoding='utf-8') as f: 
            for line in tqdm(f,desc="加载数据"):
                arr = line.strip().split("\t")
                if len(arr)<2:
                    continue

                label_id,sentence = arr
                segment = sentence.split(" ")
                
                label_id = int(label_id)
                segment_ids = [int(i) for i in segment]
                segment_ids = segment_ids[:max_seq_len]

                seq_len = len(segment_ids)
                # padding
                padding = [0]*(max_seq_len-len(segment_ids))
                segment_ids += padding

                self.data.append({
                    "label_id":torch.LongTensor([label_id]).to('cpu'),
                    "seq_len":seq_len,
                    "segment_ids":torch.LongTensor(segment_ids).to('cpu')
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
