# ELMo

## 简介

ELMo (Embeddings from Language Models) 来源于 Matthew E. Peters 等人的论文《Deep contextualized word representations》。

作者认为好的词表征模型应该解决两个问题：

1. 词语用法在语义和语法上的复杂特点。
2. 在不同语言环境下，词语的多义性。

>word2vec 中多义词都是同一个词向量，ELMo根据句子输入生成词向量，同一个词可以产生不同词向量

于是提出了deep contextualized word representation 方法来解决以上两个问题。

对于ELMo，每一个词语的表征都是整个输入语句的函数。具体做法是先在大语料上以language model为目标训练出bidirectional LSTM模型，得到LSTM产生的词语的表征（所以叫 Embeddings from Language Models）。然后利用 biLM 所有层的内部表征，这样能够产生更丰富的词语表征，高层LSTM可以捕获语境中的语义特征（可做语义消歧），底层LSTM可以捕获语法特征（可做词性标注），把他们结合在一起，用于下游的NLP任务时更能体现优势。当用于下游 nlp 任务时，先用任务的语料库（忽略 label)进行 LM 的微调，然后用label信息进行监督学习。

## 原理

EMLo 使用双向LSTM来做语言模型，目标函数是取这两个方向语言模型的最大似然

![ELMo 模型](/assets/images/bert/bert_gpt_elmo.jpg)
<center>最右侧为 ELMo 模型</center>

前向：

$$
p(t_1,t_2,\cdots,t_N)=\prod_{k=1}^N p(t_k|t_1,t_2,\cdots,t_{k-1})
$$

后向：


$$
p(t_1,t_2,\cdots,t_N)=\prod_{k=1}^N p(t_k|t_{k+1},t_{k+2},\cdots,t_N)
$$

结合两个方向的结果，最大化对数似然函数：

$$
\sum_{k=1}^N(\log p(t_k|t_1,\cdots,t_{k-1};\Theta_x,\overrightarrow{\Theta}_{LSTM},\Theta_s) \\
+\log p(t_k|t_{k+1},\cdots,t_N;\Theta_x,\overleftarrow{\Theta}_{LSTM},\Theta_s))
$$

其中$\Theta_x$是输入的词向量参数，$\Theta_s$是 Softmax 层的参数，$\overrightarrow{\Theta}_{LSTM}$和$\overleftarrow{\Theta}_{LSTM}$是多层前向和后向LSTM的参数

ELMo 对于每一个token,L层的biLM共有 $2L+1$个表征（包含最初的token词向量）

$$
\begin{aligned}
R_k &= \{x_k^{LM},\overrightarrow{h}_{k,j}^{LM},\overleftarrow{h}_{k,j}^{LM} | j=1,\cdots,L\} \\\\
 
& =\{h_{k,j}^{LM}|j=0,\cdots,L\}
\end{aligned}
$$

$h_{k,j}^{LM}$是简写，当$j=0$时，代表token层，$j>0$时，同时包含两个方向的$h$。

在下游的任务中，ELMo把所有层的表征$R$压缩在一起形成一个单独的vector。（最简单的情况下，可以值保留最后一层的$h_{k,L}^{LM}$。）

$$
ELMo_k^{task} = E(R_k;\Theta^{task}) = \gamma^{task}\sum_{i=0}^L s_j^{task}h_{k,j}^{LM}
$$

其中$s_j^{task}$是softmax归一化权重，$\gamma^{task}$是允许下游任务模型缩放 ELMo 向量的参数，它在帮助优化的过程中有实际重要性。考虑到每个biLM层的激活有不同分布，在某些情况下，对每层biLM做权重之前先进行层归一化也是有帮助的。

## 在有监督的 NLP 任务中使用 biLMs

在给点一个已经预训练好的biLM 和一个针对目标NLP任务的有监督架构后，可以很方便的使用 biLM。我们只需运行 biLM 并记录每个词的所有层的表示，然后让任务模型学习这些表示的一个线性组合。如下所述：

首先考虑无 biLM 的监督模型最底层。大多数 NLP 监督模型最底层具有共同的结构允许我们用一种一致的，统一的方式添加 ELMo。给一个 tokens 序列$(t_1,\cdots,t_N)$，每一个 token 都可以使用预训练的词嵌入和可选的字符嵌入(用单词的各个字符得到的单词的嵌入)得到token的上下文无关的表示 $x_k$。然后，模型通常会使用双向RNNs,CNNs或前馈网络得到一个上下文敏感的表示$h_k$。

为了将 ELMo 添加到监督模型中，我们首先冻结 biLM 的权重，然后连接 ELMo 向量 $ELMo_k^{task}$ 和 $x_k$。把 ELMo 增强的表示 $[x_k;ELMo_k^{task}]$ 传入任务 RNN。对于一些任务（如 SNLI,SQuAD）我们进一步观察到，在任务 RNN 的输出中引入另一组权重得到的 ELMo，效果得到了提升，即用$[h_k;ELMo_k^{task}]$代替$h_k$。由于监督模型其余部分保持不变，这些添加方式可以在更复杂的神经网络上下文中发生。

最后，我们发现在 ELMo 中适量添加 dropout，或在损失中添加 ELMo 权重的正则化 $\lambda||\mathrm{w}||^2_2$ 是有益的。给 ELMo 的权重强加一个归纳偏差来保持所有 biLM 层均值接近。

---
**参考**：
1. Matthew E. Peters 等人的论文：[Deep contextualized word representations](https://arxiv.org/abs/1802.05365)
2. [ELMo算法原理解析](https://www.cnblogs.com/huangyc/p/9860430.html)
3. [ELMo算法介绍](https://blog.csdn.net/triplemeng/article/details/82380202)
4. [文本分类实战（九）—— ELMO 预训练模型](https://www.cnblogs.com/jiangxinyang/p/10235054.html)
5. 代码：[bilm-tf](https://github.com/allenai/bilm-tf)