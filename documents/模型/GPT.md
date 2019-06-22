# GPT

GPT 是 OpenAI 团队提出的，目标是学习一个通用的表示，能够在大量任务上进行应用。结合了 transformers 和无监督学习两个已经存在的方法。项目结果证明了将监督学习和无监督预训练结合的方法十分有效。

## 原理

GPT使用无监督的预训练和有监督的微调。具体包括两个阶段。
1. 学习大语料库上的高容量语言模型。
2. 微调阶段，用带标签数据调整模型到目标任务。

## 模型结构

### 1. 无监督预训练

给出无监督语料的 tokens $\mathcal{U}=\{u_1,\cdots,u_n\}$，
用标准语言模型目标最大化似然函数：

$$
L_1(\mathcal{U}) = \sum_i \log P(u_i|u_{i-k},\cdots,u_{i-1};\Theta)
$$

其中$k$是上下文窗口大小，$\Theta$是利用神经网络对条件概率$P$建模的参数，这些参数使用随机梯度下降法训练。

文章中使用的是多层Transformer的decoder的语言模型。这个多层的结构应用multi-headed self-attention处理输入的文本，然后是 position-wise 的前馈网络，输出是词的概念分布。

$$
\begin{aligned}
h_0 &= UW_e + W_p \\\\
h_l &= transformer_block(h_{l-1}) \forall i \in [1,n] \\\\
P(u) &= softmax(h_nW_e^T)
\end{aligned}
$$

其中 $U=(u_{-k},\cdots,u_{-1})$ 是tokens的上下文向量，$n$是层数，$W_e$是token的嵌入矩阵，$W_p$是位置嵌入矩阵。


### 2. 有监督微调

这个阶段要对前一个阶段模型的参数，根据监督任务进行调整。我们假设有标签数据集$\mathcal{C}$，每一个实例由序列输入$x^1,\cdots,x^m$和标签$y$组成。经过我们预训练的模型获得输出向量$h_l^m$，然后经过参数为$W_y$的线性层来预测标签$y$。

$$
P(y|x^1,\cdots,x^m) = softmax(h_l^mW_y)
$$

下面是我们的最大化目标：

$$
L_2(\mathcal{C}) = \sum_{(x,y)} \log P(y|x^1,\cdots,x^m)
$$

我们还发现，将语言建模作为辅助目标进行微调可以提升监督模型的泛化能力及加速收敛。这与之前的观察到用辅助目标提升性能的工作相一致。具体来说,我们优化下面的目标：

$$
L_3(\mathcal{C})=L_2(\mathcal{C})+\lambda * L_1(\mathcal{C})
$$

总的来说，微调过程中我们需要的惟一额外参数是$W_y$，以及用于分隔符标记的嵌入

![gpt](/assets/images/gpt/gpt.png)


### 3. 特殊任务的输入变换

对于一些任务，比如文本分类，我们可以像上面描述的那样直接微调我们的模型。某些其他任务，如问答或文本蕴涵，有结构化的输入，如有序的句对，或文档、问题和答案的三联组。因为我们的预训练模型是训练连续序列的文本，我们需要一些修改，以适用于这些任务。以前的工作提出用顶层表征来学习任务特定结构，这种方法要求引入大量特定于任务的定制，但没有对这些额外的结构组件使用转移学习。相反，我们使用traversal-style 来解决，在这里我们将结构化的输入转换为我们预训练模型能够处理的有序句子。这些输入转换可以避免对跨任务的结构进行大量的更改。上图 Figure 1 是一个可视化说明。所有转换都包含被随机初始化的开始和结束标签($<s>,<e>$)。

文本蕴含：对于蕴含任务，我们将前提 $p$ 和假设 $h$ 的 token 序列用分隔符 `$` 连接起来。

相似度：对于相似度任务，两个句子之间没有内在的顺序。为了反映这一点，我们修改输入序列，使其包含两种可能的句子顺序(中间有一个分隔符)，并分别处理每一种顺序，以生成两个序列表征$h_l^m$。在输入线性输出层之前，按元素相加。

问答和常识推理：对于这些任务，我们有一个上下文文档 $z$，一个问题 $q$ 和一组可能的答案${a_k}$。我们把上下文文档和问题与每个可能的答案连接，在中间加上分隔符token，得到 $[z;q;\$;a_k]$,每一个序列都由模型独立处理，最后通过softmax归一化，在可能的答案上生成输出分布。


---
**参考**：
1. 论文：[Improving Language Understanding
by Generative Pre-Training](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)

2. 官方代码：[finetune-transformer-lm](https://github.com/openai/finetune-transformer-lm)

3. 官方博客：[Improving Language
Understanding with
Unsupervised Learning](https://openai.com/blog/language-unsupervised/)

4. 望江人工智库：[Improving Language Understanding by Generative Pre-Training](https://yuanxiaosc.github.io/2018/11/19/Improving_Language_Understanding_by_Generative_Pre-Training/)

5. [自然语言处理中的语言模型预训练方法（ELMo、GPT和BERT）](https://www.cnblogs.com/robert-dlut/p/9824346.html)

6. [OpenAI GPT算法原理解析](https://www.cnblogs.com/huangyc/p/9860181.html)

