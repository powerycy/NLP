# LSTM

RNN 虽然能够记住历史信息，能解决时序依赖的问题。但在实践中发现，在时序间隔增大时，RNN很难学到远距离的信息，难以解决**长期依赖问题**。而且在反向更新时存在梯度消失和梯度爆炸的问题。

LSTM（Long Short Term Memory）则解决了**长期依赖**和**梯度消失和爆炸**的问题。

LSTM 通过刻意的设计来避免长期依赖问题。记住长期的信息在实践中是 LSTM 的默认行为，而非需要付出很大代价才能获得的能力！

## 结构


![RNN](/assets/images/lstm/rnn.png)
<center>标准 RNN 中的重复模块包含单一的层</center>


![LSTM](/assets/images/lstm/lstm.png)
<center>LSTM 中的重复模块包含四个交互的层</center>

![LSTM 中的图标](/assets/images/lstm/lstm_icon.png)
<center>LSTM 中的图标</center>

## 思想

LSTM 通过三个“门结构”来控制“细胞状态”，来保证信息的传送。

细胞状态类似于传送带。直接在整个链上运行，只有一些少量的线性交互，信息在上面流传保持不变会很容易。如下图：

![细胞状态“传送带”](/assets/images/lstm/lstm_c.png)
<center>细胞状态“传送带”</center>

门结构来去除或者增加信息到细胞状态的能力。门是一种让信息选择式通过的方法。他们包含一个 sigmoid 神经网络层和一个按位的乘法操作。如下图：


![门结构](/assets/images/lstm/lstm_door.png)
<center>门结构</center>

Sigmoid 层输出 0 到 1 之间的数值，描述每个部分有多少量可以通过。0 代表“不许任何量通过”，1 就指“允许任意量通过”

### 遗忘门

决定我们会从细胞状态中丢弃什么信息

![遗忘门](/assets/images/lstm/lstm_forget.png)
<center>遗忘门</center>

### 输入门

决定什么样的新信息被存放在细胞状态中。这里包含两个部分。第一：sigmoid层，决定要更新多少值，第二：tanh层创建一个新的候选值向量

![输入门](/assets/images/lstm/lstm_input.png)
<center>输入门</center>

### 更新

更新旧细胞状态，把旧状态与$f_t$相乘，丢弃掉我们确定需要丢弃的信息。接着加上$i_t * \tilde{C}_t$。这就是新的候选值，根据我们决定更新每个状态的程度进行变化。

![更新](/assets/images/lstm/lstm_update.png)
<center>更新</center>

### 输出门

最终，我们需要确定输出什么值。这个输出将会基于我们的细胞状态，但是也是一个过滤后的版本。首先，我们运行一个 sigmoid 层来确定细胞状态的哪个部分将输出出去。接着，我们把细胞状态通过 tanh 进行处理（得到一个在  到  之间的值）并将它和 sigmoid 门的输出相乘，最终我们仅仅会输出我们确定输出的那部分。

![输出门](/assets/images/lstm/lstm_output.png)
<center>输出门</center>

### 公式总结
遗忘门：

$$
f_t = \sigma(W_f \cdot [h_{t-1},x_t]+b_f)
$$ 

输入门：

$$
\begin{aligned}
i_t &= \sigma(W_i \cdot [h_{t-1},x_t]+b_i) \\
\tilde{C_t} &= tanh(W_C \cdot [h_{t-1},x_t]+b_C) 
\end{aligned}
$$

更新：

$$
C_t = f_t \times C_{t-1} + i_t \times \tilde{C_t}
$$

输出门：

$$
\begin{aligned}
o_t &= \sigma (W_o\cdot [h_{t-1},x_t] + b_o) \\
h_t &= o_t \times tanh(C_t)
\end{aligned}
$$

## 变体

### peephole connection (窥视孔连接)

其中一个流形的 LSTM 变体，就是由 Gers & Schmidhuber (2000) 提出的，增加了 “peephole connection”。是说，我们让 门 也会接受细胞状态的输入。

![peephole connection](/assets/images/lstm/peephole_connection.png)
<center>peephole connection</center>

上面的图例中，我们增加了 peephole 到每个门上，但是许多论文会加入部分的 peephole 而非所有都加。

### coupled (耦合)

耦合输入门和忘记门。不同于之前是分开确定什么忘记和新加什么信息，这里是一同做出决定。我们仅仅会对忘记的状态位置输入新的信息。


![coupled](/assets/images/lstm/coupled.png)
<center>coupled</center>


### GRU（Gated Recurrent Unit)
由 Cho, et al. (2014) 提出。它将忘记门和输入门合成了一个单一的 更新门。同样还混合了细胞状态和隐藏状态，和其他一些改动。最终的模型比标准的 LSTM 模型要简单，也是非常流行的变体。GRU 参数更少因此更容易收敛，但是数据集很大的情况下，LSTM表达性能更好。

![GRU](/assets/images/lstm/GRU.png)
<center>GRU</center>

在GRU模型中只有两个门：分别是更新门和重置门。图中的$z_t$和$r_t$分别表示更新门和重置门。更新门用于控制前一时刻的状态信息被带入到当前状态中的程度，更新门的值越大说明前一时刻的状态信息带入越多。重置门控制前一状态有多少信息被写入到当前的候选集$\tilde{h}_t$上，重置门越小，前一状态的信息被写入的越少。

### 其他变种

当然还有很多其他的，如Yao, et al. (2015) 提出的 Depth Gated RNN。还有用一些完全不同的观点来解决长期依赖的问题，如Koutnik, et al. (2014) 提出的 Clockwork RNN。
要问哪个变体是最好的？其中的差异性真的重要吗？Greff, et al. (2015) 给出了流行变体的比较，结论是他们基本上是一样的。Jozefowicz, et al. (2015) 则在超过 1 万种 RNN 架构上进行了测试，发现一些架构在某些任务上也取得了比 LSTM 更好的结果。


## numpy 与 pytorch 对比验证 LSTM

[代码参考](/codes/rnn_examples/rnn_examples.py)

pytorch中核心部分为c++代码，整块权重按ifgo顺序划分，[代码参考](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/RNN.cpp)

tensorflow中BasicLSTMCell多一个 forget_bias 偏置项，默认为1，说明中提到加这个偏置的原因是为了在训练开始时减小遗忘的规模。因为遗忘门值越大，记住的历史信息越多。

```python
import numpy as np

import torch
from torch import nn

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
```

## numpy 与 pytorch 对比验证 GRU

[代码参考](/codes/rnn_examples/rnn_examples.py)

pytorch中核心部分为c++代码，整块权重按rzn顺序划分(重置们、更新们、候选集)，[代码参考](https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/RNN.cpp)

但代码实现与论文并不完全一致：
1. 候选集的计算，论文中是$\tilde{h}_t=\tanh(W_{\tilde{h}} \cdot [r_t*h_{t-1},x_t])$,代码中是$\tilde{h}_t=\tanh(r_t*(W_{\tilde{h}_h} \cdot h_{t-1}) + W_{\tilde{h}_x} \cdot x_t)$
2. 最终隐藏状态的计算，论文中是$(1-z_t)*h_{t-1}+z_t*\tilde{h}_t$，代码中是$z_t*h_{t-1}+(1-z_t)*\tilde{h}_t$

```python

import numpy as np

import torch
from torch import nn

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
```

---
**参考**：
1. [[译] 理解 LSTM 网络](https://www.jianshu.com/p/9dc9f41f0b29)
2. colah's blog ：[Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
3. [深度学习之GRU网络](https://www.cnblogs.com/jiangxinyang/p/9376021.html)