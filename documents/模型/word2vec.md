# word2vec

## 词向量

在NLP任务中，我们将自然语言交给机器学习算法来处理，但机器无法直接理解人类的语言，因此首先要做的事情就是将语言数学化，如何对自然语言进行数学化呢？此项来那个提供了一种很好的方式。

一种最简单的词向量是one-hot representation,就是用一个很长的向量来表示一个词，向量的长度为词典的大小N，向量的分量只有一个1，其它全为0，1的位置对应该词在词典中的索引。但是这种表示方法却有很多问题。最大的问题是我们的词汇表一般都非常大，比如达到百万级别，这样每个词都用百万维的向量来表示简直是内存的灾难。这样的向量其实除了一个位置是1，其余的位置全部都是0，表达的效率不高，而且，它也不能很好地刻画词与词之间的相似性。

另一种词向量是distributed representation，它最早是 Hinton于1986年提出的，可以克服 one-hot representation的上述缺点。其基本想法是：通过训练将某种语言中的每一个词映射成一个固定长度的短向量(当然这里的“短”是相对于 one-hot representation的“长”而言的)，所有这些向量构成一个词向量空间，而每一向量则可视为该空间中的一个点，在这个空间上引入“距离”，就可以根据词之间的距离来判断它们之间的(词法、语义上的)相似性了。比如一个常常被提到的例子：$\overrightarrow{King}-\overrightarrow{Man}+\overrightarrow{Woman}=\overrightarrow{Queen}$。
word2vec中采用的就是这种 distributed representation的词向量

下图是论文：[Exploiting Similarities among Languages for Machine Translation](https://arxiv.org/abs/1309.4168)中的一个示例：

![词向量在不同语言中的空间位置示意图](/assets/images/word2vec/词向量在不同语言中的空间位置示意图.png)

图中是讲英语和西班牙语中的词向量经过PCA降维为二维向量后在平面中画出来的。可以看到对应词汇在两个向量空间中的相对位置差不多，这说明两种不同语言对应向量空间的结构之间具有相似性，从而进一步说明了在词向量空间中利用距离刻画词之间相似性的合理性。

注意，词向量只是针对“词”来提的，事实上，我们也可以针对更细粒度或更粗粒度来进行推广，如字向量，句子向量和文档向量。

## 神经概率语言模型

在word2vec之前Bengio就已经提出了把单词表示为连续向量的神经概率语言模型。这是一个三层的神经网络来构建的语言模型，同样也是n-gram模型。如图：

![nnlm](/assets/images/word2vec/nnlm.png)

图中最下方的$w_{t-n+1},\cdots,w_{t-2},w_{t-1}$就是前n-1个词，根据这已知的$n-1$个词来预测下一个词$w_t$。$C(w)$表示词$w$所对应的词向量，整个模型中使用的是一套唯一的词向量，存在矩阵$C$（一个$|V| \times m$的矩阵）中。其中$|V|$表示词表的大小（语料中的总词数），$m$表示词向量的维度。$w$到$C(w)$的转化就是从矩阵中取出一行。

网络的第一层（输入层）是将$C(w_{t-n+1}),\cdots,C(w_{t-2},C(w_{t-1})$这 n−1 个向量首尾相接拼起来，形成一个$(n-1)m$ 维的向量，下面记为$x$。

网络的第二层（隐藏层）就如同普通的神经网络，直接使用$d+Hx$ 计算得到。$d$ 是一个偏置项。在此之后，使用 tanh 作为激活函数。

网络的第三层（输出层）一共有 $|V|$ 个节点，每个节点 $y_i$ 表示 下一个词为 $i$ 的未归一化 log 概率。最后使用 softmax 激活函数将输出值 $y$ 归一化成概率。最终，$y$ 的计算公式为：

$$y=b+Wx+U\tanh(d+Hx)$$

式子中的 $U$（一个 $|V| \times h$ 的矩阵）是隐藏层到输出层的参数，整个模型的多数计算集中在 $U$ 和隐藏层的矩阵乘法中。后面多项工作，都有对这一环节的简化，提升计算的速度。

式子中还有一个矩阵 $W$（$|V| \times (n−1)m$），这个矩阵包含了从输入层到输出层的直连边。直连边就是从输入层直接到输出层的一个线性变换(残差连接)。如果不需要直连边的话，将 $W$ 置为 0 就可以了。在最后的实验中，Bengio 发现直连边虽然不能提升模型效果，但是可以少一半的迭代次数。同时他也猜想如果没有直连边，可能可以生成更好的词向量。

现在万事俱备，用随机梯度下降法把这个模型优化出来就可以了。需要注意的是，一般神经网络的输入层只是一个输入值，而在这里，输入层 $x$ 也是参数（存在 $C$ 中），也是需要优化的。优化结束之后，词向量有了，语言模型也有了。

这样得到的语言模型自带平滑，无需传统 n-gram 模型中那些复杂的平滑算法。Bengio 在 APNews 数据集上做的对比实验也表明他的模型效果比精心设计平滑算法的普通 n-gram 算法要好 10% 到 20%。

## word2vec

word2vec作为神经概率语言模型的输入，其本身其实是神经概率模型的副产品，是为了通过神经网络学习某个语言模型而产生的中间结果。具体来说，“某个语言模型”指的是“CBOW”和“Skip-gram”。具体学习过程会用到两个降低复杂度的近似方法——Hierarchical Softmax或Negative Sampling。两个模型乘以两种方法，一共有四种实现。这些内容就是本文理论部分要详细阐明的全部了。

![word2vec](/assets/images/word2vec/word2vec.png)
<center>word2vec两种模型的示意图</center>

之前的神经概率语言模型，有三层，输入层（词向量），隐藏层和输出层（softmax层）。里面最大的问题在于从隐藏层到输出的softmax层的计算量很大，因为要计算所有词的softmax概率，再去找概率最大的值。

word2vec对这个模型做了改进，首先，对于从输入层到隐藏层的映射，没有采取神经网络的线性变换加激活函数的方法，而是采用简单的对所有输入词向量求和并取平均的方法。比如输入的是三个4维词向量：(1,2,3,4),(9,6,11,8),(5,10,7,12),那么我们word2vec映射后的词向量就是(5,6,7,8)。由于这里是从多个词向量变成了一个词向量。

第二个改进就是从隐藏层到输出的softmax层这里的计算量个改进。为了避免要计算所有词的softmax概率，word2vec采样了霍夫曼树来代替从隐藏层到输出softmax层的映射。这里就是理解word2vec的关键所在了。

### CBOW

CBOW 是 Continuous Bag-of-Words Model 的缩写，是一种根据上下文的词语预测当前词语的出现概率的模型。其图示如上图左。

CBOW是已知上下文，估算当前词语的语言模型。其学习目标是最大化对数似然函数：

$$\mathcal{L} = \sum_{w\in \mathcal{C}} \log p(w|Context(w))$$

其中，$w$表示语料库$\mathcal{C}$中任意一个词。


输入层是上下文的词语的词向量（训练开始时，词向量是随机值，随着训练不断被更新，最后成为模型的一个副产品）

投影层对其求和，所谓求和，就是简单的向量加法



---
**参考**：  
1. 论文：[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
2. 论文：[Exploiting Similarities among Languages for Machine Translation](https://arxiv.org/abs/1309.4168)
3. licstar的博客：[Deep Learning in NLP （一）词向量和语言模型](http://licstar.net/archives/328)

5. word2vec 中的数学原理详解
    * [word2vec 中的数学原理详解（一）目录和前言](https://blog.csdn.net/itplus/article/details/37969519)
    * [word2vec 中的数学原理详解（二）预备知识](https://blog.csdn.net/itplus/article/details/37969635)
    * [word2vec 中的数学原理详解（三）背景知识](https://blog.csdn.net/itplus/article/details/37969817)
    * [word2vec 中的数学原理详解（四）基于 Hierarchical Softmax 的模型](https://blog.csdn.net/itplus/article/details/37969979)
    * [word2vec 中的数学原理详解（五）基于 Negative Sampling 的模型](https://blog.csdn.net/itplus/article/details/37998797)
    * [word2vec 中的数学原理详解（六）若干源码细节](https://blog.csdn.net/itplus/article/details/37999613)
2. 刘建平博客
    * [word2vec原理(一) CBOW与Skip-Gram模型基础](https://www.cnblogs.com/pinard/p/7160330.html)
    * [word2vec原理(二) 基于Hierarchical Softmax的模型](https://www.cnblogs.com/pinard/p/7243513.html)
    * [word2vec原理(三) 基于Negative Sampling的模型](https://www.cnblogs.com/pinard/p/7249903.html)

3. [word2vec原理推导与代码分析](http://www.hankcs.com/nlp/word2vec.html)

3. [Chris McCormick](http://mccormickml.com/tutorials/)
    * [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
    * [Word2Vec Tutorial Part 2 - Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
4. [A Neural Probabilistic Language Model 论文阅读及实战](https://www.jianshu.com/p/be242ed3f314)
