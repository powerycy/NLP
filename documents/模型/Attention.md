# Attention

## 来源
Attention是一种用于提升基于RNN（LSTM或GRU）的 Encoder-Decoder 模型的效果的的机制。

《Sequence to Sequence Learning with Neural Networks》介绍了一种基于RNN的Seq2Seq模型。Encoder 把输入X编码成固定长度的隐向量Z，Decoder基于隐向量Z解码出目标输出Y。这个模型存在2个问题：
1. 把输入编码为固定长度隐向量Z，忽略了输入X的长度。当输入句子比训练集中的句子长度还长时，模型性能急剧下降。
2. 编码为固定长度隐向量Z时，句子中每个词的权重都一样，这是不合理的，在解码时没有区分度。

2015年，Dzmitry Bahdanau等人在《Neural machine translation by jointly learning to align and translate》提出了Attention Mechanism，用于对输入X的不同部分赋予不同的权重，进而实现软区分的目的。

## Attention 机制的原理

Dzmitry Bahdanau等人在论文《Neural Machine Translation by Jointly Learning to Align and Translate》中提出的 Attention 图解如下：

![Attention 图解](/assets/images/attention/attention.jpg)
<center>Attention 图解</center>

图中是模型源句子$(x_1,x_2,\cdots,x_T)$生成第$t$个目标词汇$y_t$时的示意图。

在预测第$t$个输出时，我们先利用上一步的隐状态$s_{t-1}$计算出于源句子中各个$h_j$的得分：

$$
e_{ij} = a(s_{i-1},h_j)
$$

进行softmax归一化转换为权重：

$$
\alpha_{ij} = \frac{exp(e_{ij})}{\sum_{k=1}^{T_x}exp(e_{ik})}
$$

而对所有的$h_j$的加权求和即为第$t$步解码时的背景向量$c_i$:

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij}h_j
$$

根据背景向量$c_i$和$s_{i-1}$、$y_{i-1}$，则可以得到第$t$步的隐状态：


$$
s_i = f(s_{i-1},y_{i-1},c_i)
$$

进而可得到第$t$步解码的条件概率：

$$
p(y_i|y_1,\cdots,y_{i-1},X) = g(y_{i-1},s_i,c_i)
$$

上述步骤中，$e_{ij}$即相当于一个对齐模型，用于衡量encoder端第$j$个词与decoder端第$i$个词的对齐程度（影响程度）。论文中的计算方式为：

$$
e_{ij}=v_a^{\top} \tanh(W_as_{i-1}+U_ah_j)
$$
实际上$e_{ij}$有多种计算方式，我们将在后面提到。

## Attention 机制的本质思想

把 Attention 机制从Encoder-Decoder框架中剥离，并进一步抽象，可以更容易看懂 Attention 机制的本质思想

![attention本质](/assets/images/attention/attention本质.jpg)
<center>attention本质</center>

Attention机制的实质其实就是一个寻址（addressing）的过程。将Source中的构成元素想象成是由一系列的<Key,Value>数据对构成，此时给定Target中的某个元素Query，通过计算Query和各个Key的相似性或者相关性，得到每个Key对应Value的权重系数，然后对Value进行加权求和，即得到了最终的Attention数值。也可以说Attention机制是对Source中元素的Value值进行加权求和，而Query和Key用来计算对应Value的权重系数。即可以将其本质思想改写为如下公式：

$$
Attention(Query,Source) = \sum_{i=1}^{L_x}Similarity(Query,Key_i)*Vlue_i
$$

其中$L_x=||Source||$,代表 Source 的长度

Attention机制可以看做是一种软寻址（Soft Addressing)。$Source$可以看做是存储的内容，有地址 $Key$ 和值 $Value$ 组成。根据查询输入 $Query$ 找到对应的 $Key$,取出其中的 $Value$ 值，即 $Attention$ 数值。而 $Query$ 是通过与 $Key$ 的地址进行相似性比较来寻址的，之所以说是软寻址，是因为他不像一般的寻址，只从存储内容中找出一条内容，而是可能从每个 $key$ 地址都会取出内容，取出内容的重要性根据 $Query$ 和 $Key$ 的相似性来决定，之后对 $Value$ 进行加权求和，这样就可以取出最终的 $Value$ 值，也即 $Attention$ 值。

至于 Attention 机制的具体计算过程，如果对目前大多数方法进行抽象的话，可以将其归纳为两个过程：第一个过程是根据 $Query$ 和 $Key$ 计算权重系数，第二个过程根据权重系数对 $Value$ 进行加权求和。而第一个过程又可以细分为两个阶段：第一个阶段根据 $Query$ 和 $Key$ 计算两者的相似性或者相关性；第二个阶段对第一阶段的原始分值进行归一化处理；这样，可以将Attention的计算过程抽象为如下图展示的三个阶段。

![attention计算过程](/assets/images/attention/attention计算过程.jpg)
<center>attention计算过程</center>


1. 第一阶段：计算 $Query$ 和 $Key$ 的相似性

在第一个阶段，可以引入不同的函数和计算机制，根据 $Query$ 和某个 $Key_i$，计算两者的相似性或者相关性，最常见的方法包括：

* 点积：

$$s(q,k_i) = k_i^\top q$$

* 缩放点积：

$$s(q,k_i) = \frac{k_i^\top q}{\sqrt{d}}$$

* 余弦相似度：

$$s(q,k_i) = \frac{k_i^\top q}{||q||\cdot ||k_i||}$$


* 加性（拼接）

$$
\begin{aligned}
s(q,k_i) &= V^\top \tanh (W k_i+Uq) \\
&=V^\top \tanh (W'[k_i,q])
\end{aligned}
$$

* 双线性

$$s(q,k_i) = k_i^\top Wq$$

2. 第二阶段：归一化

$$
a_i = softmax(s_i) = \frac{e^{s_i}}{\sum_{j=1}^{L_x}e^{s_j}}
$$

3. 第三阶段：加权求和

$$
Attention(Query,Source)=\sum_{i=1}^{L_x}a_i\cdot Value_i
$$

## Attention 分类

> 注意：因为下文参考了不同文章，所用图及图中的符号含义与上文并不完全一致，结合图中符号理解符号含义

1. soft Attention 和 Hard Attention

这个其实前面略微提到过。Kelvin Xu等人与2015年发表论文《Show, Attend and Tell: Neural Image Caption Generation with Visual Attention》，在Image Caption中引入了Attention，当生成第i个关于图片内容描述的词时，用Attention来关联与i个词相关的图片的区域。Kelvin Xu等人在论文中使用了两种Attention Mechanism，即Soft Attention和Hard Attention。我们之前所描述的传统的Attention Mechanism就是Soft Attention。Soft Attention是参数化的（Parameterization），因此可导，可以被嵌入到模型中去，直接训练。梯度可以经过Attention Mechanism模块，反向传播到模型其他部分。

相反，Hard Attention是一个随机的过程。Hard Attention不会选择整个encoder的输出做为其输入，Hard Attention会依概率Si来采样输入端的隐状态一部分来进行计算，而不是整个encoder的隐状态。为了实现梯度的反向传播，需要采用蒙特卡洛采样的方法来估计模块的梯度。

两种Attention Mechanism都有各自的优势，但目前更多的研究和应用还是更倾向于使用Soft Attention，因为其可以直接求导，进行梯度反向传播。

2. Global Attention 和 Local Attention

Global Attention：传统的Attention model一样。所有的hidden state都被用于计算Context vector 的权重，即变长的对齐向量$a_t$（即权重），其长度等于encoder端输入句子的长度。如下图：


![global attention](/assets/images/attention/global_attention.jpg)
<center>global attention</center>

Local Attention：Global Attention有一个明显的缺点就是，每一次，encoder端的所有hidden state都要参与计算，这样做计算开销会比较大，特别是当encoder的句子偏长，比如，一段话或者一篇文章，效率偏低。因此，为了提高效率，Local Attention应运而生。

Local Attention是一种介于Kelvin Xu所提出的Soft Attention和Hard Attention之间的一种Attention方式，即把两种方式结合起来。其结构如下图：

![local attention](/assets/images/attention/local_attention.jpg)
<center>global attention</center>


Local Attention 首先会为decoder端当前的词，预测一个source端对齐位置（aligned position)$p_t$，然后基于$p_t$选择一个窗口，用于计算背景向量$c_t$。公式如下：

$$
p_t = S\cdot sigmoid(v_p^\top \tanh(W_ph_t))
$$

其中，$S$是encoder端句子长度，$v_p$和$W_p$是模型参数。此时，对齐向量$a_t$的计算公式如下：

$$
a_t(s)=align(h_t,\bar{h}_s)exp\left(-\frac{(s-p_t)^2}{2\sigma^2}\right)
$$

总之，Global Attention和Local Attention各有优劣，在实际应用中，Global Attention应用更普遍，因为local Attention需要预测一个位置向量p，这就带来两个问题：1、当encoder句子不是很长时，相对Global Attention，计算量并没有明显减小。2、位置向量pt的预测并不非常准确，这就直接计算的到的local Attention的准确率。


3. Self Attention

Self Attention与传统的Attention机制非常的不同：传统的Attention是基于source端和target端的隐变量（hidden state）计算Attention的，得到的结果是源端的每个词与目标端每个词之间的依赖关系。但Self Attention不同，它分别在source端和target端进行，仅与source input或者target input自身相关的Self Attention，捕捉source端或target端自身的词与词之间的依赖关系；然后再把source端的得到的self Attention加入到target端得到的Attention中，捕捉source端和target端词与词之间的依赖关系。因此，self Attention Attention比传统的Attention mechanism效果要好，主要原因之一是，传统的Attention机制忽略了源端或目标端句子中词与词之间的依赖关系，相对比，self Attention可以不仅可以得到源端与目标端词与词之间的依赖关系，同时还可以有效获取源端或目标端自身词与词之间的依赖关系。如下图：


![self attention](/assets/images/attention/self_attention.jpg)
<center>self attention</center>


self attention 更相信的内容请参考[Transformer](/documents/模型/Transformer.md)，self attention 中也涉及到 Multi-Head Attention

4. Hierarchical Attention

Zichao Yang等人在论文《Hierarchical Attention Networks for Document Classification》提出了Hierarchical Attention用于文档分类。Hierarchical Attention构建了两个层次的Attention Mechanism，第一个层次是对句子中每个词的attention，即word attention；第二个层次是针对文档中每个句子的attention，即sentence attention。网络结构如下图


![分层注意力机制结构示意图](/assets/images/han/HAN.png)
<center>分层注意力机制结构示意图</center>

具体内容可参考[HAN](/documents/模型/HAN.md)

---
**参考**：
1. Dzmitry Bahdanau 等人的论文：[Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

1. [nlp中的Attention注意力机制+Transformer详解](https://zhuanlan.zhihu.com/p/53682800)
2. [模型汇总24 - 深度学习中Attention Mechanism详细介绍：原理、分类及应用](https://zhuanlan.zhihu.com/p/31547842)
3. 张俊林：[深度学习中的注意力模型（2017版）](https://zhuanlan.zhihu.com/p/37601161)