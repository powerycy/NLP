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

网络的第三层（输出层）一共有 $|V|$ 个结点，每个结点 $y_i$ 表示 下一个词为 $i$ 的未归一化 log 概率。最后使用 softmax 激活函数将输出值 $y$ 归一化成概率。最终，$y$ 的计算公式为：

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

第二个改进就是从隐藏层到输出的softmax层这里的计算量个改进。为了避免要计算所有词的softmax概率，word2vec采样了 Huffman 树来代替从隐藏层到输出softmax层的映射。这里就是理解word2vec的关键所在了。

## Hierarchical Softmax

### CBOW 模型

CBOW 是 Continuous Bag-of-Words Model 的缩写，是一种根据上下文的词语预测当前词语的出现概率的模型。其图示如上图左。

CBOW是已知上下文，估算当前词语的语言模型。其学习目标是最大化对数似然函数：

$$\mathcal{L} = \sum_{w\in \mathcal{C}} \log p(w|Context(w))$$

其中，$w$表示语料库$\mathcal{C}$中任意一个词。


输入层是上下文的词语的词向量（训练开始时，词向量是随机值，随着训练不断被更新，最后成为模型的一个副产品）

投影层对其求和，所谓求和，就是简单的向量加法。并取平均（有些资料求平均，有些没求，CS224n课程中求平均了,参考[CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/readings/cs224n-2019-notes01-wordvecs1.pdf)）。

输出层输出最可能的$w$。由于语料库中词汇量是固定的$|C|$个，所以上述过程其实可以看做是一个多分类问题。给定特征，从$|C|$个分类中挑一个。

对于神经网络模型多分类，最朴素的做法是softmax回归。
softmax回归需要对语料库中每个词语都计算一遍输出概率并进行归一化，在十几万词汇量的语料上无疑是令人头疼的。

为了优化softmax的计算，这里被替换为了一颗二叉 Huffman 树，（[关于Huffman树的知识点我](/documents/Huffman编码.md)）softmax的概率只需要沿着树形结构进行就可以了。在 Huffman 树中，隐藏层到输出层的softmax映射不是一下子完成的，而是沿着 Huffman 树一步步完成的，因此这种softmax取名为"Hierarchical Softmax"。如图：

![cbow](/assets/images/word2vec/cbow.jpg)
<center>cbow</center>

怎么沿着 Huffman 树一步步完成呢？在word2vec中，我们采用了二元逻辑回归的方法，即规定沿着左子树走，那么就是负类( Huffman 树编码1)，沿着右子树走，那么就是正类( Huffman 编码0)。判别正类和负类的方法是使用sigmoid函数。在 Huffman 树每个非叶子结点都包含一个和投影层（隐藏层）输出同样维度的参数$\theta$。词$w$的编码则是由$l-1$位编码构成（根结点不对应编码，$l$是词$w$经过的路径上的结点数）。

对于词典$\mathcal{D}$中的任意词$w$,Huffman 树中必存在一条从根结点到词$w$的路径，路径上包含$l-1$个分支，每个分支看做一次二分类，每一次分类产生一个概率，讲这些概率撑起来，就是所需的$p(w|Context(w))$

用公式表达如下（先声明公式中各参数的含义）

参数具体含义：
1. $p^w$：从根结点出发到达$w$对应叶子结点的路径
2. $l^w$：路径$p^w$中包含结点的个数
3. $p_1^w,p_2^w,\cdots,p_{l^w}^w$：路径$p^w$中的$l^w$个结点，$p_{l^w}^w$表示词$w$对应的结点
4. $d_2^w,d_3^w,\cdots,d_{l^w}^w \in \{0,1\}$：词$w$的 Huffman 编码，它由$l^w-1$位编码构成，$d_j^w$表示路径$p^w$中第j个结点对应的编码（根结点不对应编码）
5. $\theta_1^w,\theta_2^w,\cdots,\theta_{l^w-1}^w \in \mathbb{R}^m$：路径$p^w$中**非叶子结点**对应的向量，$\theta_j^w$表示路径$p^w$中第$j$个非叶子结点对应的向量


完整公式：

$$
p(w|Context(w)) = \prod_{j=2}^{l_w} p(d_j^w|X_w,\theta_{j-1}^w)
$$

其中

$$
p(d_j^w|X_w,\theta_{j-1}^w) = 
\begin{cases}
\sigma(X_w^\top\theta_{j-1}^w), \quad & d_j^w=0; \\
1-\sigma(X_w^\top\theta_{j-1}^w), \quad & d_j^w=1; 
\end{cases}
$$

或者写成整体表达式

$$

p(d_j^w|X_w,\theta_{j-1}^w) = [\sigma(X_w^\top\theta_{j-1}^w)]^{1-d_j^w}\cdot[1-\sigma(X_w^\top\theta_{j-1}^w)]^{d_j^w}
$$


则其对数似然函数为：

$$
\begin{aligned}
\mathcal{L}= & \sum_{w\in \mathcal{C}} \log \prod_{j=2}^{l^w}\{ [\sigma(X_w^\top\theta_{j-1}^w)]^{1-d_j^w}\cdot[1-\sigma(X_w^\top\theta_{j-1}^w)]^{d_j^w} \} \\
&=\sum_{w\in \mathcal{C}} \sum_{j=2}^{l^w} \{ (1-d_j^w)\cdot \log [\sigma(X_w^\top\theta_{j-1}^w)] + d_j^w \cdot \log[1-\sigma(X_w^\top\theta_{j-1}^w)]  \}
\end{aligned}
$$

这就是CBOW模型的目标函数，用随机梯度上升法更新参数$\theta_{j-1}^w$和$X_w$，最大化该目标函数即可。


### Skip-gram 模型

Skip-gram 只是逆转了CBOW的因果关系而已，即已知当前词语，预测上下文。

其网络结构如下图所示：


![skip-gram](/assets/images/word2vec/skip-gram.jpg)
<center>skip-gram</center>

上图与CBOW的两个不同在于：
1. 输入层不再是多个词向量，而是一个词向量
2. 投影层其实什么事情都没干，直接将输入层的词向量传递给输出层

在对其推导之前需要引入一个新的记号：

$u$：表示$w$的上下文中的一个词语。

于是语言模型的概率函数可以写作：

$$p(Context(w)|w)=\prod_{u\in Context(w)}p(u|w)$$

注意这是一个词袋模型，所以每个u是无序的，或者说，互相独立的。

在Hierarchical Softmax思想下，每个u都可以编码为一条01路径：

$$
p(u|w)=\prod_{j=2}^{l^u}p(d_j^u|\mathrm{v}(w),\theta_{j-1}^u)
$$

类似地，每一项都是如下简写：

$$
p(d_j^u|\mathrm{v}(w),\theta_{j-1}^u) = [\sigma(\mathrm{v}(w)^\top)\theta_{j-1}^u]^{1-d_j^u} \cdot [1-\sigma(\mathrm{v}(w)^\top\theta_{j-1}^u)]^{d_j^u}
$$

把他们写在一起，得到目标函数：

$$
\begin{aligned}
\mathcal{L} &= \sum_{w\in \mathcal{C}} \log \prod_{u\in Context(w)} \prod_{j=2}^{l^u}\{ [\sigma(\mathrm{v}(w)^\top)\theta_{j-1}^u]^{1-d_j^u} \cdot [1-\sigma(\mathrm{v}(w)^\top\theta_{j-1}^u)]^{d_j^u}\} \\

&= \sum_{w\in \mathcal{C}} \sum_{u\in Context(w)} \sum_{j=2}^{l^u} \{ (1-d_j^u)\cdot \log  [\sigma(\mathrm{v}(w)^\top)\theta_{j-1}^u] + d_j^u \cdot \log [1-\sigma(\mathrm{v}(w)^\top\theta_{j-1}^u)]\}
\end{aligned}
$$

类似CBOW的做法，将每一项简记为：

$$
\mathcal{L}(w,u,j)=(1-d_j^u)\cdot \log  [\sigma(\mathrm{v}(w)^\top)\theta_{j-1}^u] + d_j^u \cdot \log [1-\sigma(\mathrm{v}(w)^\top\theta_{j-1}^u)]
$$

虽然上式对比CBOW多了一个$u$，但给定训练实例（一个词$w$和它的上下文$\{u\}$），$u$也是固定的。所以上式其实依然只有两个变量$X_w$和$\theta_{j-1}^w$。利用随机梯度上升法最大化损失函数即可。

不过在word2vec源码中，并不是等$Context(w)$中的所有词都处理完后才刷新$\mathrm{v}(w)$，而是，每处理完$Context(w)$中的一个词$u$，就及时刷新一次$\mathrm{v}(w)$。

## Negative Sampling

Negative Sampling是Tomas Mikolov等人在[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)中提出的，它是NCE(Noise Contrastive Estimation)的一个简化版本，目的是用来提高训练速度并改善所得词向量的质量。与Hierarchical Softmax相比，NEG不再使用（复杂的）Huffman 树，而是利用（相对简单的）随机负采样，能大幅度提高性能，因而可作为Hierarchical Softmax的一种替代。[负采样算法具体内容点这里](/documents/负采样算法.md)

### CBOW 模型

在CBOW模型中，已知词$w$的上下文$Context(w)$，需要预测$w$,因此，对于给定的$Context(w)$，词$w$就是一个正样本，其它词就是负样本了。

假定现在已经选好了一个关于$w$的负样本子集$NEG(w) \neq \varnothing$。且对于$\forall \widetilde{w} \in \mathcal{D}$,定义

$$
L^w(\widetilde{w})=
\begin{cases}
1, \quad \widetilde{w} = w \\
0, \quad \widetilde{w} \neq w
\end{cases}
$$

表示词$\widetilde{w}$的标签，即正样本的标签为1，负样本的标签为0。

对于一个给定的正样本$(Context(w),w)$，我们希望最大化

$$
g(w)=\prod_{u\in \{w\} \cup NEG(w)} p(u|Context(w))
$$

其中

$$
p(u|Context(w)) = 
\begin{cases}
\sigma (\mathrm{x}_w^\top \theta^u), & \quad L^w(u)=1 \\
1-\sigma (\mathrm{x}_w^\top \theta^u), & \quad L^w(u)=0 
\end{cases}
$$

或者写成整体表达式

$$
p(u|Context(w)) = [\sigma (\mathrm{x}_w^\top \theta^u)]^{L^w(u)} \cdot [1-\sigma (\mathrm{x}_w^\top \theta^u)]^{1-L^w(u)}
$$


这里$\mathrm{x}_w$仍表示$Context(w)$中各词的词向量之和，而$\theta^u \in \mathbb{R}^m$表示词$u$对应的一个（辅助）向量，为待训练参数。

则

$$
g(w)=\sigma (\mathrm{x}_w^\top \theta^w) \prod_{u\in NEG(w)}[1-\sigma (\mathrm{x}_w^\top \theta^u)]
$$

其中

$\sigma (\mathrm{x}_w^\top \theta^w)$表示上下文为$Context(w)$，预测中心词为$w$的概率。

$\sigma (\mathrm{x}_w^\top \theta^u),u\in NEG(w)$表示上下文为$Context(w)$，预测中心词为$u$的概率。

最大化$g(w)$,相当于最大化
$\sigma (\mathrm{x}_w^\top \theta^w)$，同时最小化所有的
$\sigma (\mathrm{x}_w^\top \theta^u),u\in NEG(w)$。即增大正样本概率同时降低负样本的概率。于是，对于一个给定的语料库$\mathcal{C}$,函数

$$G=\prod_{w\in \mathcal{C}}g(w)$$

就可以作为整体优化的目标，当然，为了计算方便，对$G$取对数，最终的目标函数就是

$$
\begin{aligned}
\mathcal{L} & = \log G = \log \prod_{w\in \mathcal{C}} g(w) = \sum_{w\in \mathcal{C}} \log g(w) \\
& = \sum_{w\in \mathcal{C}} \log \prod_{u\in \{w\} \cup NEG(w)} \{ [\sigma (\mathrm{x}_w^\top \theta^u)]^{L^w(u)} \cdot [1-\sigma (\mathrm{x}_w^\top \theta^u)]^{1-L^w(u)} \} \\
& = \sum_{w\in \mathcal{C}} \sum_{u\in \{w\} \cup NEG(w)} \{ L^w(u)\cdot \log [\sigma (\mathrm{x}_w^\top \theta^u)] + [1-L^w(u)] \cdot \log [1-\sigma (\mathrm{x}_w^\top \theta^u)] \}
 
\end{aligned}
 $$

 可继续改进为：

$$
\begin{aligned}
\mathcal{L} & = \sum_{w\in \mathcal{C}} \sum_{u\in \{w\} \cup NEG(w)} \{ L^w(u)\cdot \log [\sigma (\mathrm{x}_w^\top \theta^u)] + [1-L^w(u)] \cdot \log [1-\sigma (\mathrm{x}_w^\top \theta^u)] \} \\

& = \sum_{w\in \mathcal{C}} \left\{ \log [\sigma (\mathrm{x}_w^\top \theta^w)] + \sum_{u \in NEG(w)} \log [1-\sigma (\mathrm{x}_w^\top \theta^u)] \right\} \\

& = \sum_{w\in \mathcal{C}} \left\{ \log [\sigma (\mathrm{x}_w^\top \theta^w)] + \sum_{u \in NEG(w)} \log [\sigma (-\mathrm{x}_w^\top \theta^u)] \right\} 

\end{aligned}
 $$

 之后利用随机梯度上升法最大化损失函数即可

### Skip-gram 模型

有了 Hierarchical Softmax 框架下由 CBOW 模型过渡到 Skip-gram 模型的推导经验，这里，我们仍然可以这样来做。首先，将优化目标函数由原来的

$$G=\prod_{w\in \mathcal{C}} g(w)$$

改写为：

$$G=\prod_{w\in \mathcal{C}} \prod_{u\in Context(w)} g(u)$$

这里，$\prod_{u\in Context(w)} g(u)$表示对于一个特定的样本$(w,Context(w))$，我们希望最大化的量，$g(u)$类似于上节的$g(w)$，定义为：

$$
g(u)=\prod_{z\in \{u\} \cup NEG(u)} p(z|w)
$$

其中$NEG(u)$表示处理词$u$时生成的负样本子集，条件概率

$$
p(z|w) = 
\begin{cases}
\sigma (\mathrm{v}(w)^\top \theta^z), & \quad L^u(z)=1 \\
1-\sigma (\mathrm{v}(w)^\top \theta^z), & \quad L^u(z)=0 
\end{cases}
$$

或者写成整体表达式


$$
p(z|w) = [\sigma (\mathrm{v}(w)^\top \theta^z)]^{L^u(z)} \cdot [1-\sigma (\mathrm{v}(w)^\top \theta^z)]^{1-L^u(z)}
$$

同样，我们取$G$的对数，最终的目标函数就是：

$$
\begin{aligned}

\mathcal{L} &= \log G \\
& = \log \prod_{w\in \mathcal{C}} \prod_{u\in Context(w)} g(u) \\

& = \sum_{w\in \mathcal{C}} \sum_{u\in Context(w)} \log g(u) \\
& = \sum_{w\in \mathcal{C}} \sum_{u\in Context(w)} \log \prod_{z\in \{ u\} \cup NEG(u)} p(z|w) \\
& = \sum_{w\in \mathcal{C}} \sum_{u\in Context(w)}  \sum_{z\in \{ u\} \cup NEG(u)} \log p(z|w) \\
& = \sum_{w\in \mathcal{C}} \sum_{u\in Context(w)}  \sum_{z\in \{ u\} \cup NEG(u)} \log \left\{ [\sigma (\mathrm{v}(w)^\top \theta^z)]^{L^u(z)} \cdot [1-\sigma (\mathrm{v}(w)^\top \theta^z)]^{1-L^u(z)} \right\} \\
& = \sum_{w\in \mathcal{C}} \sum_{u\in Context(w)}  \sum_{z\in \{ u\} \cup NEG(u)}  \left\{ L^u(z) \cdot \log [\sigma (\mathrm{v}(w)^\top \theta^z)] +[1-L^u(z)]\cdot \log[1-\sigma (\mathrm{v}(w)^\top \theta^z)] \right\} \\

\end{aligned}

$$

接下来梯度计算和参数更新即可。

值得一提的是，word2vec源代码中基于 Negative Sampling 的 Skip-gram 模型并不是基于目标函数来编程的，基于上式来编程的话，对于每个样本$(w,Context(w))$,需要针对$Context(w)$中的每一个词进行负采样，而word2vec源码中只是针对$w$进行了$|Context(w|$次负采样。

word2vec代码中本质上还是CBOW模型，只是将原来通过求和累加做整体用的上下文$Context(w)$拆成一个一个来考虑，此时，对于一个给定的样本$(w,Context(w))$，我们最大化的是：

$$
g(w)= \prod_{\widetilde w \in Context(w)} \prod_{u\in \{w\} \cup NEG^{\widetilde w}(w)} p(u|\widetilde w)
$$

其中

$$
p(u|\widetilde w) = 
\begin{cases}
\sigma (\mathrm{v}(\widetilde w)^\top \theta^u), & \quad L^w(u)=1 \\
1-\sigma (\mathrm{v}(\widetilde w)^\top \theta^u), & \quad L^w(u)=0 
\end{cases}
$$

或者写成整体表达式


$$
p(u|\widetilde w) = [\sigma (\mathrm{v}(\widetilde w)^\top \theta^u)]^{L^w(u)} \cdot [1-\sigma (\mathrm{v}(\widetilde w)^\top \theta^u)]^{1-L^w(u)}
$$


这里$NEG^{\widetilde w}(w)$表示处理词$\widetilde w$时生成的负样本子集。于是，对于一个给定的语料库$\mathcal{C}$,函数

$$G=\prod_{w\in \mathcal{C}} g(w)$$

就可以作为整体优化的目标。同样，我们去$G$的对数，最终的目标函数就是


$$
\begin{aligned}

\mathcal{L} &= \log G \\
& = \log \prod_{w\in \mathcal{C}}  g(w) \\

& = \sum_{w\in \mathcal{C}} \log g(w) \\
& = \sum_{w\in \mathcal{C}} \log \prod_{\widetilde w \in Context(w)} \prod_{u\in \{w\} \cup NEG^{\widetilde w}(w)} \left\{ [\sigma (\mathrm{v}(\widetilde w)^\top \theta^u)]^{L^w(u)} \cdot [1-\sigma (\mathrm{v}(\widetilde w)^\top \theta^u)]^{1-L^w(u)} \right\} \\
& = \sum_{w\in \mathcal{C}} \sum_{\widetilde w \in Context(w)} \sum_{u\in \{w\} \cup NEG^{\widetilde w}(w)} \left\{ L^w(u) \cdot \log [\sigma (\mathrm{v}(\widetilde w)^\top \theta^u)] +[1-L^w(u)]\cdot \log[1-\sigma (\mathrm{v}(\widetilde w)^\top \theta^u)] \right\} \\

\end{aligned}

$$

随机梯度上升即可


## word2vec 源码
官方代码：https://code.google.com/p/word2vec/
如果访问不了，可以访问github上的一个导入

https://github.com/tmikolov/word2vec/blob/master/word2vec.c

在源代码中，基于Negative Sampling的CBOW模型算法在464-494行，基于Hierarchical Softmax的Skip-Gram的模型算法在520-542行。大家可以对着源代码再深入研究下算法。

另外，vocab[word].code[d]指的是，当前单词word的，第d个编码，编码不含Root结点。vocab[word].point[d]指的是，当前单词word，第d个编码下，前置的结点。这些和基于Hierarchical Softmax的是一样的。

## gensim 中 word2vec 包的使用

[代码位置](/codes/word2vec/word2vec_examples.py)

拿到了原文，我们首先要进行分词。加入下面的一串人名是为了结巴分词能更准确的把人名分出来。
```python
import jieba
from gensim.models import word2vec

# 分词
jieba.suggest_freq('沙瑞金', True)
jieba.suggest_freq('田国富', True)
jieba.suggest_freq('高育良', True)
jieba.suggest_freq('侯亮平', True)
jieba.suggest_freq('钟小艾', True)
jieba.suggest_freq('陈岩石', True)
jieba.suggest_freq('欧阳菁', True)
jieba.suggest_freq('易学习', True)
jieba.suggest_freq('王大路', True)
jieba.suggest_freq('蔡成功', True)
jieba.suggest_freq('孙连城', True)
jieba.suggest_freq('季昌明', True)
jieba.suggest_freq('丁义珍', True)
jieba.suggest_freq('郑西坡', True)
jieba.suggest_freq('赵东来', True)
jieba.suggest_freq('高小琴', True)
jieba.suggest_freq('赵瑞龙', True)
jieba.suggest_freq('林华华', True)
jieba.suggest_freq('陆亦可', True)
jieba.suggest_freq('刘新建', True)
jieba.suggest_freq('刘庆祝', True)

with open('./in_the_name_of_people.txt','r',encoding='utf-8') as rf,\
    open('./in_the_name_of_people_segment.txt','w',encoding='utf-8') as wf:
    
    document = rf.read()
    document_cut = jieba.cut(document)
    result = " ".join(document_cut)

    wf.write(result)
```

拿到了分词后的文件，在一般的NLP处理中，会需要去停用词。由于word2vec的算法依赖于上下文，而上下文有可能就是停词。因此对于word2vec，我们可以不用去停词。这里只是一个示例，因此省去了调参的步骤，实际使用的时候，你可能需要对我们上面提到一些参数进行调参。

```python
# 训练模型
sentences = word2vec.LineSentence('./in_the_name_of_people_segment.txt')
model = word2vec.Word2Vec(sentences, hs=1,min_count=1,window=3,size=100)
```

使用训练好的word2vec模型

1. 找出某个词向量最相近的词集合

```python
# 查看模型效果
print("-"*10,"5个与'李达康'相近的3个字的词：","-"*10)
req_count = 5
for key in model.wv.similar_by_word("李达康",topn=100):
    if len(key[0])==3:
        req_count -=1
        print(key)
        if req_count == 0:
            break

print("-"*10,"5个与'陆亦可'相近的3个字的词：","-"*10)
req_count = 5
for key in model.wv.similar_by_word("陆亦可",topn=100):
    if len(key[0])==3:
        req_count -=1
        print(key)
        if req_count == 0:
            break
```

```
---------- 5个与'李达康'相近的3个字的词： ----------
('侯亮平', 0.9734761714935303)
('陆亦可', 0.9610140323638916)
('祁同伟', 0.9597763419151306)
('钟小艾', 0.9588832855224609)
('赵东来', 0.9577993750572205)
---------- 5个与'陆亦可'相近的3个字的词： ----------
('侯亮平', 0.971320390701294)
('郑西坡', 0.964660108089447)
('祁同伟', 0.9637094736099243)
('李达康', 0.9610140323638916)
('陈岩石', 0.960942268371582)
```      

2. 查看两个词向量的相似程度
```python
print("-"*10,"两个词之间的相似度","-"*10)
print("沙瑞金 高育良：",model.wv.similarity("沙瑞金","高育良"))
print("高育良 沙瑞金：",model.wv.similarity("高育良","沙瑞金"))
print("李达康 侯亮平：",model.wv.similarity("李达康","侯亮平"))
```

```
---------- 两个词之间的相似度 ----------
沙瑞金 高育良： 0.9759979781032404
高育良 沙瑞金： 0.9759979781032404
李达康 侯亮平： 0.9734761557161287
```

3. 找出不同类的词

```python
print("-"*10,"给定列表中的哪个单词与其他单词不一致","-"*10)
print("沙瑞金 高育良 李达康 侯亮平：",model.wv.doesnt_match("沙瑞金 高育良 李达康 侯亮平".split()))
print("沙瑞金 高育良 李达康 刘庆祝：",model.wv.doesnt_match("沙瑞金 高育良 李达康 刘庆祝".split()))
```

```
---------- 给定列表中的哪个单词与其他单词不一致 ----------
沙瑞金 高育良 李达康 侯亮平： 李达康
沙瑞金 高育良 李达康 刘庆祝： 刘庆祝
```

模型的保存于加载

```python
# 保存模型，保存词向量,加载模型
# model.save("./word2vec_gensim")
# model.wv.save_word2vec_format("data/model/word2vec_org",
#                                   "data/model/vocabulary",
#                                   binary=False)
# model = word2vec.Word2Vec.load('./word2vec_gensim')
```

算法中有关的参数都在类gensim.models.word2vec.Word2Vec中。

算法需要注意的参数有：

1. sentences: 我们要分析的语料，可以是一个列表，或者从文件中遍历读出。

2. size: 词向量的维度，默认值是100。这个维度的取值一般与我们的语料的大小相关，如果是不大的语料，比如小于100M的文本语料，则使用默认值一般就可以了。如果是超大的语料，建议增大维度。

3. window：即当前词与预测词的最大距离，意味着上下文共2*window个词。window越大，则和某一词较远的词也会产生上下文关系。默认值为5。在实际使用中，可以根据实际的需求来动态调整这个window的大小。如果是小语料则这个值可以设的更小。对于一般的语料这个值推荐在[5,10]之间。

4. sg: 即我们的word2vec两个模型的选择了。如果是0， 则是CBOW模型，是1则是Skip-Gram模型，默认是0即CBOW模型。

5. hs: 即我们的word2vec两个解法的选择了，如果是0， 则是Negative Sampling，是1的话并且负采样个数negative大于0， 则是Hierarchical Softmax。默认是0即Negative Sampling。

6. negative:即使用Negative Sampling时负采样的个数，默认是5。推荐在[3,10]之间。

7. cbow_mean: 仅用于CBOW在做投影的时候，为0，则算法中的xw为上下文的词向量之和，为1则为上下文的词向量的平均值。在我们的原理篇中，是按照词向量的平均值来描述的。个人比较喜欢用平均值来表示xw,默认值也是1,不推荐修改默认值。

8. min_count:需要计算词向量的最小词频。这个值可以去掉一些很生僻的低频词，默认是5。如果是小语料，可以调低这个值。

9. iter: 随机梯度下降法中迭代的最大次数，默认是5。对于大语料，可以增大这个值。

10. alpha: 在随机梯度下降法中迭代的初始步长。算法原理篇中标记为η，默认是0.025。

11. min_alpha: 由于算法支持在迭代的过程中逐渐减小步长，min_alpha给出了最小的迭代步长值。随机梯度下降中每轮的迭代步长可以由iter，alpha， min_alpha一起得出。这部分由于不是word2vec算法的核心内容，因此在原理篇我们没有提到。对于大语料，需要对alpha, min_alpha,iter一起调参，来选择合适的三个值。


---
**参考**：  
1. 论文：[Efficient Estimation of Word Representations in Vector Space](https://arxiv.org/abs/1301.3781)
2. 论文：[Exploiting Similarities among Languages for Machine Translation](https://arxiv.org/abs/1309.4168)
3. 论文：[Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546)
4. licstar的博客：[Deep Learning in NLP （一）词向量和语言模型](http://licstar.net/archives/328)

5. word2vec 中的数学原理详解
    * [word2vec 中的数学原理详解（一）目录和前言](https://blog.csdn.net/itplus/article/details/37969519)
    * [word2vec 中的数学原理详解（二）预备知识](https://blog.csdn.net/itplus/article/details/37969635)
    * [word2vec 中的数学原理详解（三）背景知识](https://blog.csdn.net/itplus/article/details/37969817)
    * [word2vec 中的数学原理详解（四）基于 Hierarchical Softmax 的模型](https://blog.csdn.net/itplus/article/details/37969979)
    * [word2vec 中的数学原理详解（五）基于 Negative Sampling 的模型](https://blog.csdn.net/itplus/article/details/37998797)
    * [word2vec 中的数学原理详解（六）若干源码细节](https://blog.csdn.net/itplus/article/details/37999613)
6. 刘建平博客
    * [word2vec原理(一) CBOW与Skip-Gram模型基础](https://www.cnblogs.com/pinard/p/7160330.html)
    * [word2vec原理(二) 基于Hierarchical Softmax的模型](https://www.cnblogs.com/pinard/p/7243513.html)
    * [word2vec原理(三) 基于Negative Sampling的模型](https://www.cnblogs.com/pinard/p/7249903.html)
    * [用gensim学习word2vec](https://www.cnblogs.com/pinard/p/7278324.html)

7. [word2vec原理推导与代码分析](http://www.hankcs.com/nlp/word2vec.html)

8. [Chris McCormick](http://mccormickml.com/tutorials/)
    * [Word2Vec Tutorial - The Skip-Gram Model](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
    * [Word2Vec Tutorial Part 2 - Negative Sampling](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
9. [A Neural Probabilistic Language Model 论文阅读及实战](https://www.jianshu.com/p/be242ed3f314)
