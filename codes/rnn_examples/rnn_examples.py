#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-25 16:32:49
@author: wind
'''
import numpy as np

import torch
from torch import nn

##### 用 numpy 验证 pytorch 中的 RNNCell 的计算过程
print("验证 RNNCell")
N = 5 # 样本数
D = 10 # 样本维度
H = 7 # 隐藏单元维度

# 输入
x = np.random.random((N,D)).astype(np.float32)
h0 = np.random.random((N,H)).astype(np.float32)

# RNNCell结果
rnn = nn.RNNCell(D,H)
h_torch = rnn(torch.tensor(x),torch.tensor(h0))
h_torch = h_torch.data.numpy()

# 获取个权重
w_hh = rnn.weight_hh.data.numpy().T
w_ih = rnn.weight_ih.data.numpy().T
b_hh = rnn.bias_hh.data.numpy().T
b_ih = rnn.bias_ih.data.numpy().T

# numpy 结果
h_np = np.tanh(np.dot(x,w_ih) + b_ih + np.dot(h0,w_hh) + b_hh)

# 对比结果
print("h_hp == h_torch :",(np.abs(h_np-h_torch<1e-6)).all())

#===== 结果
# 验证 RNNCell
# h_hp == h_torch : True


##### 用 numpy 验证 pytorch 中的 LSTMCell 的计算过程
print("验证 LSTMCell")
def sigmoid(x):
    return 1/(1+np.exp(-x))

N = 5 # 样本数
D = 10 # 样本维度
H = 7 # 隐藏单元维度

# 输入
x = np.random.random((N,D)).astype(np.float32)
h0 = np.random.random((N,H)).astype(np.float32)
c0 = np.random.random((N,H)).astype(np.float32)

# LSTMCell结果
lstm = nn.LSTMCell(D,H)

h_torch,c_torch = lstm(torch.tensor(x),(torch.tensor(h0),torch.tensor(c0)))
h_torch = h_torch.data.numpy()
c_torch = c_torch.data.numpy()

# 获取个权重（遗忘门、输入门、输入值、输出门 4个权重包含在了一起）
w_hh = lstm.weight_hh.data.numpy().T
w_ih = lstm.weight_ih.data.numpy().T
b_hh = lstm.bias_hh.data.numpy().T
b_ih = lstm.bias_ih.data.numpy().T

# numpy 结果 方式一：4个权重作为整体计算
print("方式一：4个权重作为整体计算")
line_result = np.dot(x,w_ih) + b_ih + np.dot(h0,w_hh) + b_hh
iv,fv,gv,ov = np.split(line_result,4,1)

i = sigmoid(iv)
f = sigmoid(fv)
g = np.tanh(gv)
o = sigmoid(ov)

c_np = f*c0 + i*g
h_np = o*np.tanh(c_np)

# 对比结果
print("c_np == c_torch :",(np.abs(c_np-c_torch)<1e-6).all())
print("h_np == h_torch :",(np.abs(h_np-h_torch)<1e-6).all())

# numpy 结果 方式二：4个权重分开各自与x,h计算 
print("方式二：4个权重分开各自与x,h计算")
# 从权重集合中提取出各个具体权重
w_h_i,w_h_f,w_h_g,w_h_o = np.split(w_hh,4,1) # 各结果shape H*H
w_i_i,w_i_f,w_i_g,w_i_o = np.split(w_ih,4,1) # 各结果shape N*H
b_h_i,b_h_f,b_h_g,b_h_o = np.split(b_hh,4,0) # 各结果shape H
b_i_i,b_i_f,b_i_g,b_i_o = np.split(b_ih,4,0) # 各结果shape H

# numpy 结果
i = sigmoid(np.dot(x,w_i_i) + b_i_i + np.dot(h0,w_h_i) + b_h_i) # 输入门 结果shape N*H
f = sigmoid(np.dot(x,w_i_f) + b_i_f + np.dot(h0,w_h_f) + b_h_f) # 遗忘门 结果shape N*H
g = np.tanh(np.dot(x,w_i_g) + b_i_g + np.dot(h0,w_h_g) + b_h_g) # 输入值 结果shape N*H
o = sigmoid(np.dot(x,w_i_o) + b_i_o + np.dot(h0,w_h_o) + b_h_o) # 输出门 结果shape N*H

c_np = f*c0 + i*g
h_np = o*np.tanh(c_np)

# 对比结果
print("c_np == c_torch :",(np.abs(c_np-c_torch)<1e-6).all())
print("h_np == h_torch :",(np.abs(h_np-h_torch)<1e-6).all())

# numpy 结果 方式三：按理论公式，x,h做拼接，然后计算，但权重重新拆分拼接才对应公式，这里简单话，直接作为整体来算
print("方式三：按理论公式，x,h做拼接，然后计算，但权重重新拆分拼接才对应公式，这里简单话，直接作为整体来算")
w = np.vstack((w_hh,w_ih)) # 结果shape D*4H

hx = np.hstack((h0,x))
line_result = np.dot(hx,w)+b_hh+b_ih

iv,fv,gv,ov = np.split(line_result,4,1)

i = sigmoid(iv)
f = sigmoid(fv)
g = np.tanh(gv)
o = sigmoid(ov)

c_np = f*c0 + i*g
h_np = o*np.tanh(c_np)

# 对比结果
print("c_np == c_torch :",(np.abs(c_np-c_torch)<1e-6).all())
print("h_np == h_torch :",(np.abs(h_np-h_torch)<1e-6).all())

#===== 结果
# 验证 LSTMCell
# 方式一：4个权重作为整体计算
# c_np == c_torch : True
# h_np == h_torch : True
# 方式二：4个权重分开各自与x,h计算
# c_np == c_torch : True
# h_np == h_torch : True
# 方式三：按理论公式，x,h做拼接，然后计算，但权重重新拆分拼接才对应公式，这里简单话，直接作为整体来算
# c_np == c_torch : True
# h_np == h_torch : True


##### 用 numpy 验证 pytorch 中的 GRUCell 的计算过程
print("验证 GRUCell")
def sigmoid(x):
    return 1/(1+np.exp(-x))

N = 5 # 样本数
D = 10 # 样本维度
H = 7 # 隐藏单元维度

# 输入
x = np.random.random((N,D)).astype(np.float32)
h0 = np.random.random((N,H)).astype(np.float32)

# GRUCell结果
gru = nn.GRUCell(D,H)

h_torch = gru(torch.tensor(x),torch.tensor(h0))
h_torch = h_torch.data.numpy()

# 获取个权重（重置们，更新们，新信息 3个权重包含在了一起）
w_hh = gru.weight_hh.data.numpy().T
w_ih = gru.weight_ih.data.numpy().T
b_hh = gru.bias_hh.data.numpy().T
b_ih = gru.bias_ih.data.numpy().T

w_h_r,w_h_z,w_h_n = np.split(w_hh,3,1) # 各结果shape H*H
w_i_r,w_i_z,w_i_n = np.split(w_ih,3,1) # 各结果shape N*H
b_h_r,b_h_z,b_h_n = np.split(b_hh,3,0) # 各结果shape H
b_i_r,b_i_z,b_i_n = np.split(b_ih,3,0) # 各结果shape H

# numpy 结果
r = sigmoid(np.dot(x,w_i_r) + b_i_r + np.dot(h0,w_h_r) + b_h_r) # 重置们 结果shape N*H
z = sigmoid(np.dot(x,w_i_z) + b_i_z + np.dot(h0,w_h_z) + b_h_z) # 更新们 结果shape N*H
# 这里的新信息的处理和论文不太一致，论文是h0先乘以权重，再做线性变换。 这里是先做了线性变换再乘以权重
n = np.tanh(np.dot(x,w_i_n) + b_i_n + r*(np.dot(h0,w_h_n) + b_h_n)) # 新信息 结果shape N*H

h_np = z*h0 + (1-z)*n # 此处和GRU论文刚好反了，论文是 (1-z)*h0 + z*n

# 对比结果
print("h_np == h_torch :",(np.abs(h_np-h_torch)<1e-6).all())

#===== 结果
# 验证 GRUCell
# h_np == h_torch : True