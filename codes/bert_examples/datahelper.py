#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-24 18:36:07
@author: wind
'''

import os
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from pytorch_pretrained_bert import BertTokenizer


class NewsDataset(Dataset):
    def __init__(self,data_file,max_seq_len):
        super().__init__()

        self.data = []

        if os.path.isfile("./bert/bert-base-chinese-vocab.txt"):
            tokenizer = BertTokenizer.from_pretrained('./bert/bert-base-chinese-vocab.txt')
        else:
            tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
        tokenizer.max_len = 1e12

        cls_id = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
        sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
        
        with open(data_file,'r',encoding='utf-8') as f: 
            for line in tqdm(f,desc="加载数据"):
                arr = line.strip().split("\t")
                
                label_id,sentence = arr
                # label
                label_id = int(label_id)
                # 
                word_ids = [int(item) for item in sentence.split(" ")]

                word_ids = [cls_id] + word_ids[:max_seq_len-2] + [sep_id]
                segment_ids = [0] * len(word_ids)
                word_mask = [1]*len(word_ids)
                
                # padding
                padding = [0]*(max_seq_len-len(word_ids))
                word_ids += padding
                segment_ids += padding
                word_mask += padding

                self.data.append({
                    "word_ids":torch.LongTensor(word_ids),
                    "segment_ids":torch.LongTensor(segment_ids),
                    "word_mask":torch.LongTensor(word_mask),
                    "label_id":torch.LongTensor([label_id])
                })
    def __getitem__(self,item):
        return self.data[item]
    
    def __len__(self):
        return len(self.data)

def get_labels(src_file):
    with open(src_file,'r',encoding='utf-8') as f:
        return [line.strip() for line in f.readlines()]
        

if __name__ == "__main__":
    ds = NewsDataset("./data/cnews_final_test.txt",3000)
    print(ds[0])
    print(get_labels('./data/label'))
