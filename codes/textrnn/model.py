#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-23 22:29:35
@author: wind
'''
import math
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn import CrossEntropyLoss
import yaml

class Config(object):
    def __init__(self,config_file):
        with open(config_file,'r',encoding='utf-8') as f:
            cf = yaml.load(f)
        for key,value in cf.items():
            self.__dict__[key] = value

class TextRNN(nn.Module):
    def __init__(self,config,embedding_matrix):
        super().__init__()
        
        # 预训练的词向量矩阵
        self.embedding_matrix = embedding_matrix

        # Embedding
        self.embeddings = nn.Embedding(config.vocab_size,config.embedding_dim)
        # lstm
        self.lstm = nn.LSTM(
            input_size = config.embedding_dim,
            hidden_size = config.hidden_size,
            num_layers = config.num_layers,
            dropout = config.dropout,
            bidirectional = config.bidirectional
        )
        # 分类
        self.classifier = nn.Linear(config.hidden_size*(2 if config.bidirectional else 1),config.num_labels)
        
        # 初始化权重
        self.apply(self.init_weights)

    def forward(self,x,seq_lens,labels=None):
        # embedding
        em = self.embeddings(x)
        # 压缩
        packed_input = pack_padded_sequence(em, seq_lens)
        # lstm
        packed_output,(hn,cn) = self.lstm(packed_input) 

        # # 解压缩
        # output, _ = pad_packed_sequence(packed_output)
        # 不需要解压缩，hn就是每个句子最后一个字符的输出，
        # 如果解压后，去output最后的值，则是填充后最后的输出，有填充的结果就是0了

        # 不压缩
        # out:  (seq_len, batch, num_directions * hidden_size)
        # hn:  (num_layers * num_directions, batch, hidden_size)
        # cn:  (num_layers * num_directions, batch, hidden_size)
        # 压缩的话
        # out 是 PackedSequence，分2部分，一部分为压缩的数据，一部分为各序列长

        # 拆分h_n多各层，各方向
        _,batch, hidden_size = hn.size()
        hn = hn.view(self.lstm.num_layers, -1, batch, hidden_size)
        # 取最后一层结果
        hn = hn[-1]
        # 转换为 batch优先,并合
        hn = hn.transpose(0,1).view(batch,-1)

        # 分类
        logits = self.classifier(hn)

        if labels is None:
            return logits
        else:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return loss
    
    def get_labels(self,x,seq_lens):
        logits = self.forward(x,seq_lens)
        labels = torch.argmax(logits,1)
        labels = labels.to("cpu").numpy()
        return labels

    def get_loss_acc(self,x,seq_lens,labels):
        logits = self.forward(x,seq_lens)

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
        elif isinstance(module,nn.Linear):
            module.weight.data.normal_(0,0.1)
            if module.bias is not None:
                module.bias.data.zero_()
        
if __name__ == "__main__":
    import numpy as np
    config = Config('./config.yaml')
    model = TextRNN(config,None)

    x = torch.tensor(np.array([[3,5,6,7,8,9,4],
        [13,25,46,57,58,79,0],
        [563,65,46,67,78,0,0],
        [35,54,65,76,0,0,0],
        [43,65,76,0,0,0,0]]))
    x = x.transpose(0,1)

    logits = model(x,[7,6,5,4,3])
    print(logits)
