#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-23 22:29:35
@author: wind
'''
import math
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import yaml

class Config(object):
    def __init__(self,config_file):
        with open(config_file,'r',encoding='utf-8') as f:
            cf = yaml.load(f)
        for key,value in cf.items():
            self.__dict__[key] = value

class ConvItem(nn.Module):
    def __init__(self,input_channels,output_channels,kernel_size,k_max_pooling):
        super().__init__()

        self.conv = nn.Conv2d(input_channels,output_channels,kernel_size) 
        self.k_max_pooling = nn.AdaptiveMaxPool2d(k_max_pooling)
        self.relu = nn.ReLU()

    def forward(self,x):
        x = self.conv(x) # 卷积 NCHW
        x = self.relu(x) # relu激活
        x = self.k_max_pooling(x) # k_max_pooling
        return x

class TextCNN(nn.Module):
    def __init__(self,config,embedding_matrix):
        super().__init__()
        
        # 预训练的词向量矩阵
        self.embedding_matrix = embedding_matrix

        # multichannel Embedding
        self.embeddings_static = nn.Embedding(config.vocab_size,config.embedding_dim)
        self.embeddings_non_static = nn.Embedding(config.vocab_size,config.embedding_dim)
        
        # 锁定静态的Embedding，使其不进行更新
        for p in self.embeddings_static.parameters():
            p.requires_grad = False

        # 各个卷积模型
        self.conv_items = nn.ModuleList([
            ConvItem(2,
                conv["out_channels"],
                (conv["kernel_size"],config.embedding_dim),
                (conv["k_max_pooling"],1))
            for conv in config.convs])
        
        # dropout
        self.dropout = nn.Dropout(config.dropout)

        # 最后的分类层，final_dim是模型分类前的维度
        final_dim =  sum([conv["out_channels"]*conv["k_max_pooling"] for conv in config.convs])
        self.classifier = nn.Linear(final_dim,config.num_labels)
        
        # 初始化权重
        self.apply(self.init_weights)

    def forward(self,x,labels=None):
        # embedding
        em1 = self.embeddings_static(x) # NL -> NLD
        em2 = self.embeddings_non_static(x) # NL -> NLD

        # 将两个embedding合并成2个channels
        em1 = em1.unsqueeze(1) # NLD -> NCHW
        em2 = em2.unsqueeze(1) # NLD -> NCHW
        em = torch.cat((em1,em2),1) # NLHW

        # 各个卷积对2通道的embedding进行处理
        r = []
        for conv_item in self.conv_items:
            r.append(conv_item(em))
        
        # 合并多个卷积的结果
        x = torch.cat(r,2)
        # 拉平
        x = x.view(x.size(0),-1)
        # dropout
        x = self.dropout(x)
        # 分类
        logits = self.classifier(x)

        if labels is None:
            return logits
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss

    def get_labels(self,x):
        logits = self.forward(x)
        labels = torch.argmax(logits,1)
        labels = labels.to("cpu").numpy()
        return labels

    def get_loss_acc(self,x,labels):
        logits = self.forward(x)

        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits, labels)

        eq = torch.argmax(logits,1) == labels
        eq = eq.to("cpu").numpy()
        acc = eq.sum()/len(eq)

        return loss,acc

    def init_weights(self,module):
        if isinstance(module,nn.Embedding):
            if self.embedding_matrix is not None:
                module.weight.data.copy_(self.embedding_matrix)
            else:
                module.weight.data.normal_(0,0.1)
        elif isinstance(module,nn.Conv2d):
            n = module.kernel_size[0]*module.kernel_size[1]*module.out_channels
            module.weight.data.normal_(mean=0.0,std=math.sqrt(2./n))
        elif isinstance(module,nn.Linear):
            module.weight.data.normal_(0,0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        
if __name__ == "__main__":

    config = Config('./data/config.yaml')
    model = TextCNN(config,None)

    x = torch.randint(0,20,(5,76)) # N,C
    logits = model(x)
    print(logits)
    
    