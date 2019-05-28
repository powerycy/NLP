# HAN 

HAN（Hierarchical Attention Networks）来自于论文[《Hierarchical Attention Networks for Document Classification》](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)。是用于进行文档分类的模型，结合了双向RNN与注意力机制，对词级别和句子级别进行分层处理。

## 思想

主要思想是，首先考虑文档的分层结构：单词构成句子，句子构成文档，所以建模时也分这两部分进行。其次，不同的单词和句子具有不同的信息量，不能单纯的统一对待所以引入Attention机制。而且引入Attention机制除了提高模型的精确度之外还可以进行单词、句子重要性的分析和可视化，让我们对文本分类的内部有一定了解。

## 特点
1. 分层机制
2. 注意力机制

## 结构

HAN 可分为四部分：

1. 词序列的编码
2. 词序列的注意力
3. 句子序列的编码
4. 句子序列的注意力

如下图：

![Hierarchical Attention Networks](/assets/images/han/HAN.png)
<center>Hierarchical Attention Networks</center>

输入的一个文档包含$s_i$个句子，每个句子包含$T_i$个单词。
$w_{it}$代表第$i$个句子的第$t$个单词。其中$t \in [1,T]$。

### 1. 词序列的编码

单词$w_{it}$经过词嵌入矩阵$W_e$变为词向量$x_{it}$，词向量$x_{it}$经过双向GRU得到正向隐状态$\overrightarrow{h}_{it}$和反向隐状态$\overleftarrow{h}_{it}$，两个隐状态的拼接为单词$w_{it}$的表征，即$h_{it} = [\overrightarrow{h}_{it},\overleftarrow{h}_{it}]$

$$
\begin{aligned}
x_{it} &= W_ew_{it},t\in [1,T] \\
\overrightarrow{h}_{it} &= \overrightarrow{GRU}(x_{it}),t\in [1,T] \\
\overleftarrow{h}_{it} &= \overleftarrow{GRU}(x_{it}),t\in [T,1] \\
h_{it} &= [\overrightarrow{h}_{it},\overleftarrow{h}_{it}]
\end{aligned}
$$

### 2. 词序列的注意力

并不是所有的单词对句子意思的表达都有同等的贡献。所以采用了注意力机制来获得整个句向量。对$h_it$做一层变换得到$u_{it}$,然后用$u_{it}$与词级上下文向量$u_w$的相似度来衡量词的重要性,并归一化得到权重$\alpha_{it}$。然后依据权重$\alpha_{it}$和隐向量$h_{it}$得到句子向量$s_i$。其中词词上下文向量$u_{w}$是随机初始化并在训练过程中共同学习得到的（个人觉得这样很奇怪，所以自己在代码实现时，是根据隐状态$h_it$进行变换来得到的，这个变换的权重是可学习的）

$$
\begin{aligned}
u_{it} &= \tanh (W_wh_{it}+b_w) \\
\alpha_{it} &= \frac{\exp(u_{it}^\top u_w)}{\sum_t \exp(u_{it}^\top u_w)} \\
s_i &= \sum_t \alpha_{it}h_{it}

\end{aligned}
$$

### 3. 句子序列的编码

得到句向量$s_i$后，同样的方法得到文档向量

$$
\begin{aligned}
\overrightarrow{h}_{i} &= \overrightarrow{GRU}(s_{i}),t\in [1,L] \\
\overleftarrow{h}_{i} &= \overleftarrow{GRU}(x_{i}),t\in [L,1] \\
h_{i} &= [\overrightarrow{h}_{i},\overleftarrow{h}_{i}]
\end{aligned}
$$

### 4. 句子序列的注意力

$u_s$为句子级上下文向量，类似$u_w$

$$
\begin{aligned}
u_{i} &= \tanh (W_sh_{i}+b_s) \\
\alpha_{i} &= \frac{\exp(u_{i}^\top u_s)}{\sum_t \exp(u_{i}^\top u_s)} \\
v_i &= \sum_i \alpha_{i}h_{i}

\end{aligned}
$$

最后进行softmax分类

$$
p = softmax(W_cv + b_c).
$$


---
**参考**： 
1. 论文：[《Hierarchical Attention Networks for Document Classification》](https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf)
2. [Hierarchical Attention Network for Document Classification阅读笔记](https://blog.csdn.net/liuchonge/article/details/73610734)