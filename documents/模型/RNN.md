# RNN

RNN (Recurrent Neural Network)，递归神经网络。用来解决具有序列依赖的问题。

## 结构

![RNN](/assets/images/rnn/rnn_1.jpg)
![RNN](/assets/images/rnn/rnn.jpg)

* 圆圈和方块表示向量
* 一个箭头表示对该向量做一次变换

$$
h_1 = f(Ux_1+Wh_0+b) \quad y_1 = Softmax(Vh_1+c) \\
h_2 = f(Ux_2+Wh_1+b) \quad y_2 = Softmax(Vh_2+c) \\
h_3 = f(Ux_3+Wh_2+b) \quad y_3 = Softmax(Vh_3+c) \\
h_4 = f(Ux_4+Wh_2+b) \quad y_4 = Softmax(Vh_4+c) 
$$

每一步的参数$U,W,b,V,c$都一样，是共享的。  
$x_1,\cdots,x_n$ 是输入  
$y_1,\cdots,y_n$是输出  
$h_0,\cdots,h_n$是隐藏状态  
初始隐藏状态$h_0$一般是人为设定的，常用0向量初始化$h_0$



我们说递归神经网络拥有记忆能力，而这种能力就是通过W将以往的输入状态进行总结，而作为下次输入的辅助。可以这样理解隐藏状态: h=f(现有的输入+过去记忆总结)


## 使用方式


![RNN使用方式](/assets/images/rnn/rnn使用方式.jpeg)

### 1 vs N

![1vn_1](/assets/images/rnn/1vn_1.jpg)
<center>只在序列开始进行输入计算</center>

![1vn_2](/assets/images/rnn/1vn_2.jpg)
<center>把输入信息X作为每个阶段的输入</center>

![1vn_3](/assets/images/rnn/1vn_3.jpg)
<center>把输入信息X作为每个阶段的输入(简略表示)</center>

### N vs M

N vs M。这种结构又叫Encoder-Decoder模型，也可以称之为Seq2Seq模型。

原始的N vs N RNN要求序列等长，然而我们遇到的大部分问题序列都是不等长的，如机器翻译中，源语言和目标语言的句子往往并没有相同的长度。

为此，Encoder-Decoder结构先将输入数据编码成一个上下文向量c：

![N vs M 编码](/assets/images/rnn/NvsM编码.jpg)
<center>N vs M 编码</center>

编码为c有多种方式，比如：
1. 把Encoder最后一个隐状态复制给c，$c=h_4$
2. 对最后隐状态变换得到c，$c=q(h_4)$
3. 对所有隐状态做变换得到c，$c=q(h_1,h_2,h_3,h_4)$

拿到c之后，就用另一个RNN网络对其进行解码，这部分RNN网络被称为Decoder。具体做法也有多种：
1. 把c当做初始状态$h_0$输入Decoder
2. 把c作为每一步输入

如下图：

![c作为Decoder初始隐状态](/assets/images/rnn/c_as_h0.jpg)
<center>c作为Decoder初始隐状态</center>


![c作为Decoder的输入](/assets/images/rnn/c_as_input.jpg)
<center>c作为Decoder的输入</center>

## numpy 与 pytorch 对比验证

[代码参考](/codes/rnn_examples/rnn_examples.py)

```python
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
```


---
**参考**：
1. 何之源 知乎：[完全图解RNN、RNN变体、Seq2Seq、Attention机制](https://zhuanlan.zhihu.com/p/28054589)
2. Andrej Karpathy 博客：[The Unreasonable Effectiveness of Recurrent Neural Networks](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)