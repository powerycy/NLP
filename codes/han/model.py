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

class Attention(nn.Module):
    def __init__(self,hidden_size):
        super().__init__()
        # 论文中的 query 即上下文向量，说是类似记忆网络中的用法，然后随机初始化，
        # 大部分人都是直接随机初始化的，个人感觉太随意了，所以用输入的隐状态经过变换来得到
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()
    
    def forward(self,hidden_states,lengths):
        # 论文中是随机初始化的一个向量，这里个人处理成根据各个输入的隐状态变换求平均得到
        query = torch.mean(self.query(hidden_states),dim=0,keepdim=True) # (L,N,D) -> (1,N,D)
        key = self.tanh(self.key(hidden_states)) # (L,N,D)
        query = query.permute(1,2,0) # (N,D,1)
        key = key.permute(1,0,2) # (N,L,D)

        # 计算注意力分值
        attention_scores = torch.matmul(key, query) # (N,L,1)
        attention_scores = attention_scores.squeeze(-1) # (N,L)
        # padding的数据不应该被关注，所以将分值设为一个默认的很低的值
        # 有数据的地方为标记为0，padding的地方标记为-10000
        # 加到原始的分值上即可
        temp_idx = torch.arange(attention_scores.size(1)).expand_as(attention_scores)
        len_expand = lengths.unsqueeze(1).expand_as(attention_scores)
        mask = torch.ge(temp_idx,len_expand).to(dtype=attention_scores.dtype) * -10000.0
        
        attention_scores = attention_scores + mask # (N,L)

        attention_probs = nn.Softmax(dim=-1)(attention_scores) # (N,L)
        result = torch.matmul(attention_probs.unsqueeze(1), key).squeeze(1) # K和V都是同一个东西  (N,D)

        return result

class TextHAN(nn.Module):
    def __init__(self,cf,embedding_matrix):
        super().__init__()
        
        # 预训练的词向量矩阵
        self.embedding_matrix = embedding_matrix

        # Embedding
        self.embeddings = nn.Embedding(cf.vocab_size,cf.embedding_dim)
        # word_gru
        self.word_gru = nn.GRU(
            input_size = cf.embedding_dim,
            hidden_size = cf.hidden_size,
            num_layers = cf.num_layers,
            dropout = cf.dropout,
            bidirectional = cf.bidirectional
        )
        # word_attention
        self.word_attention = Attention(2*cf.hidden_size if cf.bidirectional else cf.hidden_size)
        # sent_gru 词的结果双向拼接，维度翻倍了，
        self.sent_gru = nn.GRU(
            input_size = 2*cf.hidden_size if cf.bidirectional else cf.hidden_size,
            hidden_size = 2*cf.hidden_size if cf.bidirectional else cf.hidden_size,
            num_layers = cf.num_layers,
            dropout = cf.dropout,
            bidirectional = cf.bidirectional
        )
        # sent_attention 句子的结果双向拼接，维度再次翻倍
        self.sent_attention = Attention(4*cf.hidden_size if cf.bidirectional else cf.hidden_size)
        # 分类
        self.classifier = nn.Linear(4*cf.hidden_size if cf.bidirectional else cf.hidden_size,
                                    cf.num_labels)
        
        # 初始化权重
        self.apply(self.init_weights)

    def forward(self,x,word_lens,labels=None):
        # embedding
        em = self.embeddings(x) # (n,sl,wl,d)
        n,sl,wl,d = em.size()

        # 将batch个文档的所有句子全部合在一起作为batch,
        # 有可能出现有些句子完全为空的情况，因为每个文档补全足够句子数，补的句子其实全为空
        word_x = em.view(n*sl,wl,d)
        word_x_lens = word_lens.view(n*sl)

        # 将序列按长度降序排列
        word_x_lens,word_x_idx = word_x_lens.sort(0,descending=True)
        word_x = word_x[word_x_idx]
        # 空句子无法被处理，取出长度大于0的句子
        flag = word_x_lens>0
        word_x_post = word_x[flag]
        word_x_lens_post = word_x_lens[flag]
        # 转化为 batch_first = False 的格式
        word_x_post = word_x_post.transpose(0,1) 

        # 传入GRU前的压缩
        word_packed = pack_padded_sequence(word_x_post, word_x_lens_post)
        # gru 
        word_packed_output,_ = self.word_gru(word_packed) 
        # 解压
        word_output,_ = pad_packed_sequence(word_packed_output) # (wl,sl_non_0,d)
        
        # 进行 attention，得到每个句字的向量
        sentence_vector = self.word_attention(word_output,word_x_lens_post)
        # 补上删掉的0长度句子
        n_0 = n*sl-len(sentence_vector)
        sentence_vector = torch.cat((sentence_vector,torch.zeros((n_0,)+sentence_vector.size()[1:])),dim=0)

        # 把句子还原为原来 batch 个文章的格式
        _,restore_idx = word_x_idx.sort(0,descending=False)
        sentence_vector = sentence_vector[restore_idx]

        # batch 个文档的句向量数据
        sent_x = sentence_vector.view(n,sl,-1) 
        # 每篇文档的句子个数
        sent_x_lens = word_lens.gt(0).sum(dim=1)
        # 按文档长短降序排序
        sent_x_lens,sent_x_idx = sent_x_lens.sort(0,descending=True)
        sent_x = sent_x[sent_x_idx]
        # 文档里没有空文档，终于不用考虑没数据的问题了，哈哈，继续
        # 转化为 batch_first = False 的格式
        sent_x = sent_x.transpose(0,1) 

        # 传入GRU前的压缩
        sent_packed = pack_padded_sequence(sent_x, sent_x_lens)
        # gru 
        sent_packed_output,_ = self.sent_gru(sent_packed) 
        # 解压
        sent_output,_ = pad_packed_sequence(sent_packed_output) # (sl,n,d)
        
        # 进行 attention，得到文档向量
        document_vector = self.sent_attention(sent_output,sent_x_lens)
        
        # 分类
        logits = self.classifier(document_vector)

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
    config.max_word_len = 7
    config.max_sen_len = 4
    model = TextHAN(config,None)

    x = torch.tensor([[[3,5,6,7,8,9,4],[3,5,8,9,4,0,0],[3,5,7,8,9,4,0],[0,0,0,0,0,0,0]],
        [[13,25,58,79,0,0,0],[13,57,58,79,0,0,0],[13,25,46,79,0,0,0],[13,25,46,58,79,0,0]],
        [[563,65,46,78,0,0,0],[563,65,46,67,78,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]],
        [[35,54,65,76,78,35,6],[35,54,65,76,78,35,6],[35,54,65,76,78,35,6],[0,0,0,0,0,0,0]],
        [[43,65,76,0,0,0,0],[43,65,76,23,0,0,0],[43,65,764,23,0,0,0],[43,65,35,54,65,76,0]]])
    word_lens = torch.tensor([[7,5,6,0],[4,4,4,5],[4,5,0,0],[7,7,7,0],[3,4,4,6]])

    logits = model(x,word_lens)
    print(logits)
