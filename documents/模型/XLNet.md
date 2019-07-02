# XLNet（翻译）

XLNet: Generalized Autoregressive Pretraining for Language Understanding，用于语言理解的广义自回归预训练模型

## 摘要

与基于自回归语言建模的预训练方法相比，基于降噪自编码的预训练方法（如BERT),具有更好的双向上下文建模能力。然而，由于输入中有掩码，BERT 忽略了掩码位置之间的依赖关系，并且有预训练和微调之间的差异。针对这些优缺点，我们提出了 XLNet,一种广义的自回归预训练方法，（1）通过最大化因子分解顺序的所有排列的期望似然来实现双向上下文的学习，（2）通过自回归公式克服了 BERT 的局限性。XLNet 还将最先进的自回归模型 Transformer-XL 的思想集成到预训练中。从经验上看，XLNet 在20项任务上的表现要优于 BERT，并且在问答、自然语言推理、情感分析和文档排序等 18 项任务上取得了最先进的成果。

## 1. 介绍 

无监督表征学习在自然语言处理领域取得了巨大的成功。通常，这些方法首先在大规模无标记文本语料库上对神经网络进行预训练，然后对下游任务的模型或表征进行微调。在这一共同的高层次理念下，不同的无监督预训练目标在文献中得到了探索。其中，自回归(AR,autoregressive)和自编码(AE,autoencoding)语言建模是两个最成功的预训练目标。

AR语言建模是利用自回归模型估计文本语料库的概率分布。具体来说，给定一个文本序列 $\mathrm{x}=(x_1,\cdots,x_T)$，AR 语言建模将这种似然因式分解为前向乘积 $p(\mathrm{x})=\prod_{t=1}^T p(x_t | \mathrm{x}_{<t})$ 或 后向乘积 $p(\mathrm{x})=\prod_{t=T}^1 p(x_t | \mathrm{x}_{>t})$。一个参数模型(如神经网络)被训练来对每个条件分布建模。由于AR语言模型仅被训练为编码单向上下文(前向或后向)，因此它在建模深层双向上下文时并不有效。相反，下游语言理解任务通常需要双向上下文信息。这导致了AR语言建模和有效的预训练之间存在一定差距。

相比之下，基于 AE 的预训练并没有进行显式的密度估计，而是从损坏的输入中重建原始数据。一个著名的例子是BERT，它是最先进的预训练方法。给定输入的 token 序列，用一个特殊符号 [MASK] 替换 tokens 中的特定部分，并训练模型，从损坏的版本中恢复原始 token。由于密度估计不是目标的一部分，BERT 可以利用双向上下文进行重建。作为一个直接的好处，就是消除了前面提到的 AR 语言建模中的双向信息的鸿沟，从而提高了性能。然而，BERT 在预训练中使用的[MASK]等人工符号在微调阶段并没有出现在真实的数据中，导致了预训练与微调的不一致。此外，由于预测的 tokens 在输入中是被遮蔽的，所以，BERT 不能像 AR 语言建模那样使用乘积规则对联合概率进行建模。换句话说，BERT 假设了要预测的tokens 和每个给定的未遮蔽的 token 是彼此独立的，这被过分简化了，因为在自然语言中高阶（high-order,感觉翻译为高度有序更合适）、远程依赖普遍存在。

针对现有语言预训练目标的优缺点，本文提出了 XLNet 自回归方法，该方法充分利用了 AR 语言建模和 AE 的优点，同时避免了它们的局限性。

* 首先，XLNet 不像传统 AR 模型那样使用固定的前向或后向分解顺序，而是最大化了**所有可能的因子分解顺序的排列**（all possible permutations of the factorization order）的序列的对数似然。多亏了排列操作，每个位置的上下文都可以由来自左右两边的 token 组成。在期望中，每个位置都要学会利用来自所有位置的上下文信息，即，捕获双向上下文。

* 其次，作为一种广义的 AR 语言模型，XLNet 不依赖于数据破坏。因此，XLNet 不受 BERT 那样的预训练微调差异的影响。同时，自回归目标也提供了一种自然的方法来使用乘积规则分解预测 token 的联合概率，消除了 BERT 中做出的独立性假设

除了一个新的预训练目标，XLNet 还改进了用于预训练的体系结构设计。

* 受 AR 语言建模最新进展的启发，XLNet 将 Transformer-XL 的段递归机制和相关编码方案集成到预训练中，特别是在涉及较长文本序列的任务中，提高了性能。

* 天真地将 Transformer(-XL) 体系结构应用于基于排列（ permutation-based ）的语言建模是行不通的，因为因子分解顺序是任意的，目标是模糊的。作为一种解决方案，我们建议重新参数化 Transformer(-XL) 网络，以消除歧义。

从经验上看，XLNet 在18个任务上取得了最先进的结果，即，7个GLUE语言理解任务，3个阅读理解任务(包括 SQuAD 和 RACE)，7个文本分类任务(包括Yelp 和 IMDB)，以及 ClueWeb09-B 文档排序任务。在一组公平的比较实验中，XLNet 在多个基准上始终优于 BERT。

**相关工作** 

基于排列的 AR 建模思想已经在《The Journal of Machine Learning Research》 和《International Conference on Machine Learning, 》中得到了探索，但是有几个关键的区别。以前的模型是无序的，而 XLNet 本质上是通过位置编码感知顺序的。这对于语言理解是很重要的，因为无秩序的模型会退化为 bag-of-words，缺乏基本的表达能力。上述差异源于动机的根本差异——以前的模型旨在通过在模型中引入“orderless”的归纳偏差（归纳偏差可以理解为前提假设的意思）来改进密度估计，而XLNet的动机则是让AR语言模型能够学习双向上下文。

## 2. 建议的方法

### 2.1 背景

在本节中，我们首先回顾和比较了用于语言预训练的传统 AR 语言建模和BERT。给定文本序列$\mathrm{x}=[x_1,\cdots,x_T]$，AR 语言建模在前向自回归因子分解（the forwar autoregressive factorization）下，通过最大化似然进行预训练:

$$
\tag{1}
\mathop{\max}_{\theta} \log p_\theta(\mathrm{x}) = \sum_{t=1}^T \log p_\theta (x_t| \mathrm{x}_{<t}) = \sum_{t=1}^T \log \frac{\exp(h_\theta(\mathrm{x}_{1:t-1})^\top e(x_t))}{\sum_{x'} \exp(h_\theta(\mathrm{x}_{1:t-1})^\top e(x'))}
$$

其中 $h_\theta(\mathrm{x}_{1:t-1})$ 是由神经网络产生的上下文表征，$e(x)$ 表示 $x$ 的嵌入。而 BERT 是基于降噪自动编码的（denoising auto-encoding）。具体地说，对于一个文本序列 $\mathrm{x}$，BERT 首先随机把 $\mathrm{x}$ 中一定比例（如15%）的 tokens 设置为一个特殊符号 [MASK],来构造为一个损坏的 $\hat{\mathrm{x}}$,把被遮蔽的 tokens 记为 $\bar{\mathrm{x}}$,训练目标是从$\hat{\mathrm{x}}$ 中重构 $\bar{\mathrm{x}}$：


$$
\tag{2}
\mathop{\max}_{\theta} \log p_\theta(\bar{\mathrm{x}}|\hat{\mathrm{x}}) \approx \sum_{t=1}^T m_t \log p_\theta (x_t| \hat{\mathrm{x}}) = \sum_{t=1}^T m_t \log \frac{\exp(H_\theta(\hat{\mathrm{x}})_t^\top e(x_t))}{\sum_{x'} \exp(H_\theta(\hat{\mathrm{x}})_t^\top e(x'))}
$$

其中 $m_t=1$ 表示 $x_t$ 是被遮蔽的，$H_\theta$ 是 Transformer，把长度为 $T$ 的文本序列 $\mathrm{x}$ 映射为隐藏向量序列 $H_\theta(\mathrm{x})=[H_\theta(\mathrm{x})_1,H_\theta(\mathrm{x})_2,\cdots,H_\theta(\mathrm{x})_T]$。以下几个方面比较了这两种预训练目标的优缺点：

* **独立假设**:公式(2)中用$\approx$强调，BERT 因子分解的联合条件概率 $p(\bar{\mathrm{x}}|\hat{\mathrm{x}})$ 基于这样的独立假设，即所有被遮蔽的 tokens $\bar{\mathrm{x}}$ 是分别重构的。相比之下，基于 AR 的语言建模目标，即公式(1)，使用乘法规则因子分解 $p_\theta(\mathrm{x})$，普遍没有这样一个独立假设。

* **噪音输入（Input noise）**：BERT的输入包含像[MASK]这样的人工符号，这些符号在下游任务中从未出现，这就造成了预训练微调（pretrain-finetune）差异。在 BERT 论文中，用原始 token 替换 [MASK] 并不能解决这个问题，因为原始 tokens 只能在很小的概率下使用——否则公式（2）优化起来就很简单（trivial）了。相比之下，AR 语言建模不依赖于任何输入损坏，也就不存在这个问题。

* **上下文依赖**：AR 表征 $h_\theta(\mathrm{x}_{1:t-1})$ 只需要一直到位置 $t$ 的 tokens 为条件（即左令牌），而 BERT 表征 $H_\theta(\mathrm{x})_t$ 有权访问双向上下文信息。因此，BERT 目标允许更好地捕获双向上下文来对模型进行预训练。

### 2.2 目标：排列语言建模（Permutation Language Modeling）

![排列语言模型](/assets/images/xlnet/排列语言模型.png)
图1：给定相同输入序列$\mathrm{x}$，但因子分解顺序不同的情况下预测$x_3$的排列语言建模目标示意图

通过以上比较，AR语言建模和BERT具有各自独特的优势。一个自然要问的问题是，是否存在一种既能带来双方优势，又能避免各自弱点的预训练目标。

借鉴无序 NADE（The Journal of Machine Learning Research, 17(1):7184–
7220, 2016.）的思想，我们提出了排列语言建模目标，它不仅保留了AR模型的优点，而且允许模型捕获双向上下文。具体来说，对于长度为 $T$ 的序列 $\mathrm{x}$，有$T!$ 个不同的排序来执行有效的自回归因子分解。直观地说，如果模型参数在所有因子分解顺序中共享，那么预期模型将学会从两边的所有位置收集信息。

为了形式化这个概念，让$\mathcal{Z}_T$为长度为$T$的索引序列$[1,2,\cdots,T]$的所有可能排列的集合。我们使用 $\mathcal{z}_t$ 和 $\mathrm{z}_{<t}$ 表示排列 $\mathrm{z} \in \mathcal{Z}_T$ 的第 $t$ 个元素和 之前的 $t-1$ 个元素。那么，我们提出的排列语言建模目标可以表示为:


$$
\tag{3}
\mathop{\max}_{\theta} \mathbb{E}_{\mathrm{z} \thicksim \mathcal{Z}_T }[ \sum_{t=1}^T \log p_\theta(x_{z_t}|\mathrm{x}_{\mathrm{z}<t}) ] 
$$

基本上,一个文本序列 $\mathrm{x}$,我们采样一次因子分解顺序$\mathrm{z}$,并且根据因子分解顺序分解似然函数 $p_\theta(\mathrm{x})$。因为在训练期间，所有因子分解顺序共享相同的模型参数 $\theta$，我们期望，$x_t$ 看到序列中每一个可能的元素 $x_i \neq x_t$,由此能够捕获上相上下文。此外，这个目标符合 AR 框架，它自然地避免了第2.1节中讨论的独立假设和预训练微调差异

**排列讨论** 我们提出的目标仅仅变更了因子分解顺序，而不影响序列顺序。换句话说，我们保持了原始的序列顺序，用和原序列相对应的位置编码，并通过 Transformer 中特有的注意力遮蔽来实现因子分解顺序的排列。注意，这个选择是必要的，因为模型在微调阶段只会遇到具有自然顺序的文本序列。

为了提供一个整体的图片，我们在图1中展示了一个例子，它在给定相同的输入序列$\mathrm{x}$，但是在不同的因子分解顺序下，预测$x_3$。

### 2.3 体系结构:用于目标感知表征的双流自注意力(Two-Stream Self-Attention for Target-Aware Representations)


![双流自注意力](/assets/images/xlnet/双流自注意力.png)
图2：（a）：内容流注意力，这与标准的自注意力一样。（b）：Query 流注意力，没有访问关于内容 $x_{z_t}$ 的信息。（c）：双流注意力排列语言建模训练概述

虽然排列语言建模目标具有所需的属性，但是使用标准 Transformer 参数化的简单实现可能无法工作。来看这个问题，假设我们使用标准的 Softmax 规则参数化 next-token 的分布 $p_\theta(X_{z_t}| \mathrm{x}_{\mathrm{z}<t})$，也就是 $p_\theta(X_{z_t}=x|\mathrm{X}_{\mathrm{z}<t})=\frac{\exp(e(x)^\top h_\theta(\mathrm{x}_{\mathrm{z}<t}))}{\sum_{x'} \exp(e(x')^\top h_\theta(\mathrm{x}_{\mathrm{z}<t}))}$,其中 $h_\theta(\mathrm{x}_{\mathrm{z}<t})$ 表示共享的 Transformer 网络经过适当 masking 后产生的 $\mathrm{x}_{\mathrm{z}<t}$的隐藏表征。现在注意，表征 $h_\theta(\mathrm{x}_{\mathrm{z}<t})$ 并不依赖于哪个位置的预测，即，$z_t$的值。因此，无论目标位置如何，预测的分布都是相同的，因此无法学习有用的表征（具体示例见附录 A.1）。为了避免这个问题，我们建议重新参数化 next-token 分布，使其能够感知目标位置：

$$
\tag{4}
p_\theta(X_{z_t}=x|\mathrm{X}_{\mathrm{z}<t})=\frac{\exp(e(x)^\top g_\theta(\mathrm{x}_{\mathrm{z}<t},z_t))}{\sum_{x'} \exp(e(x')^\top g_\theta(\mathrm{x}_{\mathrm{z}<t},z_t))}
$$

其中 $g_\theta(\mathrm{x}_{\mathrm{z}<t},z_t)$表示以目标位置作为另外一个输入的一种新的表征。

**双流自注意力** 目标感知表征的想法消除了在目标预测中的歧义，如何制定 $g_\theta(\mathrm{x}_{\mathrm{z}<t},z_t)$ 仍然是一个不简单的问题。在其他可能性中，我们建议“站”在目标位置 $z_t$ 并依靠位置 $z_t$ 通过注意上下文 $\mathrm{x}_{\mathrm{z}<t}$ 来收集信息。这个参数化工作，在标准 Transformer 架构中有两个要求是相互矛盾的：（1）在预测$x_{z_t}$时，$g_\theta(\mathrm{x}_{\mathrm{z}<t},z_t)$ 应该只使用位置 $z_t$，而不用内容 $x_{z_t}$，否则目标就变得微不足道了；（就是说，要预测$x_{z_t}$,而输入中如果还包含$x_{z_t}$，那还要模型干啥）（2）预测其他$x_{z_j}$,其中$j>t$时，$g_\theta(\mathrm{x}_{\mathrm{z}<t},z_t)$ 也应该编码内容 $x_{z_t}$ 以提供完整的上下文信息。为了解决这一矛盾，我们建议使用两组隐藏表征代替以前的一组表征：

* 内容表征（content representation） $h_\theta(\mathrm{x}_{\mathrm{z}\le t})$,或缩写为 $h_{z_t}$,这是和标准 Transformer 类似的隐状态。这个表征同时编码了上下文和 $x_{z_t}$自己。
* 查询表征（query representation）$g_\theta(\mathrm{x}_{\mathrm{z}<t},z_t)$,或简写为 $g_{z_t}$,只访问上下文信息 $\mathrm{x}_{\mathrm{z}<t}$ 和位置信息 $z_t$,而没有访问上面讨论的内容 $x_{z_t}$。

计算上，第一层查询流（query stream）初始化为可训练向量,即 $g_i^{(0)}=w$,内容流（content stream）设置为相应的单词嵌入，即 $h_i^{(0)}=e(x_i)$。对于每个自注意力层 $m=1,\cdots,M$,这两个表征流是按照图示更新的，使用共享的一组参数进行更新。（如图2（a）和（b）所示）：

$$
g_{z_t}^{(m)} \leftarrow \mathrm{Attention}(\mathrm{Q}=g_{z_t}^{(m-1)},\mathrm{KV}=\mathrm{h}_{\mathrm{z}<t}^{(m-1)};\theta)
$$
(query 流，使用$z_t$，但不能看到 $x_{z_t}$)

$$
h_{z_t}^{(m)} \leftarrow \mathrm{Attention}(\mathrm{Q}=h_{z_t}^{(m-1)},\mathrm{KV}=\mathrm{h}_{\mathrm{z} \le t}^{(m-1)};\theta)
$$
(content 流，同时使用$z_t$ 和 $x_{z_t}$)

其中 $\mathrm{Q},\mathrm{K},\mathrm{V}$ 表示注意力操作中的 query,key 和 value。 内容表征的更新规则与标准的 self-attention 完全相同，因此在微调期间，我们可以简单地删除 query 流，并将 content 流用于普通的 Transformer(-XL)。最后，我们可以使用最后一层 query 表征 $g_{z_t}^{(M)}$ 来计算公式 (4)

**部分预测** 虽然排列语言建模目标（公式3）有很多优点，但由于排列的存在，使得优化问题更具挑战性，在初步实验中收敛速度较慢。为了降低优化的难度，我们选择只预测因子分解顺序中的最后一个 token。形式上，我们把 $\mathrm{z}$ 分解为一个非目标子序列 $\mathrm{z}_{\le c}$ 和一个目标子序列 $\mathrm{z}_{> c}$，其中 $c$ 为切分点。目标是在非目标子序列的条件下，最大化目标子序列的对数似然，即：

$$
\tag{5}

\mathop{\max}_{\theta} \mathbb{E}_{\mathrm{z} \thicksim \mathcal{Z}_T }[ \log p_\theta(\mathrm{x}_{\mathrm{z}>c}|\mathrm{x}_{\mathrm{z} \le c})] = \mathbb{E}_{\mathrm{z} \thicksim \mathcal{Z}_T }[\sum_{t=c+1}^{|\mathrm{z}|} \log p_\theta(x_{z_t}|\mathrm{x}_{\mathrm{z}<t})]

$$

注意，选择 $\mathrm{z}_{>c}$ 作为目标是因为在给定当前因子分解顺序$\mathrm{z}$时，它拥有最长的上下文。一个超参数 $K$ 被使用，使大约 $1/K$ 的 tokens 被选择作为预测；也就是，$|\mathrm{z}|/(|\mathrm{z}|-c) \approx K$。对于未被选择的 tokens，他们的 query 表征不需要计算，节省了速度和内存。

### 2.4 整合来自 Transformer-XL 的想法

由于我们的目标函数符合 AR 框架，我们将最先进的 AR 语言模型 Transformer-XL 合并到我们的预训练框架中，并以它命名我们的方法。我们集成了 Transformer-XL 中两种重要的技术，即相对位置编码方案和段递归机制。我们对前面讨论的原始序列应用相对位置编码，这很简单。现在我们讨论如何将递归机制集成到所提议的排列设置中，并使模型能够重用之前片段中的隐藏状态。在不失一般性的前提下，假设我们从一个长序列$\mathrm{s}$中取出两个片段；即，$\tilde{\mathrm{x}}=\mathrm{s}_{1:T}$ 和 $\mathrm{x}=\mathrm{s}_{T+1:2T}$，让 $\tilde{\mathrm{z}}$ 和 $\mathrm{z}$ 分别为 $[1\cdots T]$ 和 $[T+1 \cdots 2T]$ 的排列。然后，基于排列 $\tilde{\mathrm{x}}=\mathrm{s}_{1:T}$ ，我们处理第一段，然后缓存获得每一层(用$m$表示)的 content 表征 $\tilde{\mathrm{h}}^{(m)}$。然后，对于下一段 $\mathrm{x}$，用 memory 更新注意力，可以被写成：

$$
h_{z_t}^{(m)} \leftarrow \mathrm{Attention}(\mathrm{Q}=h_{z_t}^{(m-1)},\mathrm{KV}=[\tilde{h}^{m-1},h_{\mathrm{z}_{\le t}}^{(m-1)}];\theta)
$$

其中 $[.,.]$表示沿着序列（sequence）维度的拼接。注意，位置编码只依赖于原始序列中的实际位置。因此,一旦获取表征 $\tilde{\mathrm{h}}^{(m)}$,上述注意力更新则独立于$\tilde{\mathrm{z}}$。这允许缓存和重用 memory，而不需要知道前一段的因子分解顺序。预期中,该模型学会了在最后一段的所有因子分解顺序上利用内存。可以用同样的方法计算 query 流。最后，图2（c）给出了提议的双流注意力排列语言建模的概述（参见附录A.4，以获得更详细的说明）。

### 2.5 建模多段（Modeling Multiple Segments）

许多下游任务有多个输入段，例如，在回答问题时，一个问题和一个上下文段落。现在我们讨论在自回归框架中如何预训练 XLNet 为多个段建模。在训练前阶段，按照 BERT，我们随机抽取两个片段(来自相同上下文或不同上下文)，并将两个片段连接，作为一个序列来执行排列语言建模。我们只重用属于相同上下文的 memory。具体来说，我们模型的输入类似于 BERT:[A,SEP,B,SEP,CLS],其中“SEP”和“CLS”是两个特殊符号，A和B是两个片段。虽然我们采用的是两段数据格式，但 XLNet-Large 并没有使用下句预测（next sentence）的目标，因为它没有显示出我们消融研究的持续改进(见第3.7节)。

**相对段编码** 在架构上，与BERT将绝对段嵌入添加到每个位置的单词嵌入不同，我们将相对编码的概念从 Transformer-XL 也扩展到对段进行编码。给定序列中的一对位置 $i$ 和 $j$，如果 $i$ 和 $j$ 来自同一段，我们使用一个段编码 $\mathrm{s}_{ij} = \mathrm{s}_+$或$\mathrm{s}_{ij} = \mathrm{s}_-$，其中 $\mathrm{s}_+$ 和 $\mathrm{s}_−$ 是每个注意头的可学习模型参数。换句话说，我们只考虑这两个位置是否在同一段内，而不考虑它们来自哪个特定的段。这与相对编码的核心思想是一致的;即,只建模位置之间的关系。当$i$注意到$j$时，使用片段编码$\mathrm{s}_{ij}$来计算注意力权重 $a_{ij} = (\mathrm{q}_i + \mathrm{b})^\top \mathrm{s}_{ij}$，其中$\mathrm{q}_i$为标准注意力操作中的查询向量，$\mathrm{b}$为可学习的头部特定偏置向量。最后，将$a_{ij}$的值添加到正常注意力权重中。使用相对段编码有两个好处。首先，相对编码的归纳偏差提高了泛化。其次，它打开了对具有两个以上输入段的任务进行微调的可能性，而使用绝对段编码是不可能做到这一点的。

### 2.6 讨论和分析

#### 2.6.1 和 BERT 比较

对比公式(2)和(5)，我们发现 BERT 和 XLNet 都进行了部分预测，即，只预测序列中的 tokens 子集。对于BERT来说，这是一个必要的选择，因为如果所有 tokens 都被遮蔽，就不可能做出任何有意义的预测。此外，对于 BERT 和 XLNet 来说，局部预测仅通过预测具有足够上下文的 tokens 来降低优化难度。然而，第2.1节中讨论的独立性假设使 BERT 无法对目标之间的依赖关系建模。

为了更好地理解这种差异，让我们考虑一个具体的例子[New, York, is, a, city]。假设BERT和XLNet都选择两个tokens [New, York] 作为预测目标，并最大化 $\log p(\mathrm{New \; York} \; | \; \mathrm{is \; a \; city})$。并假设 XLNet 采样的因子分解顺序为 [is, a, city, New, York]。在这个例子中，BERT 和 XLNet 分别归结为以下目标:

$$
\mathcal{J}_{\mathrm{BERT}} = \log p(\mathrm{New} \; | \; \mathrm{is \; a \; city}) + \log p(\mathrm{York} \; | \; \mathrm{is \; a \; city}) 
$$
$$
\mathcal{J}_{\mathrm{XLNet}} = \log p(\mathrm{New} \; | \; \mathrm{is \; a \; city}) + \log p(\mathrm{York} \; | \mathrm{New} ,\; \mathrm{is \; a \; city}) 
$$
请注意，XLNet 能够捕获这一对(New, York)之间的依赖关系，BERT忽略了这一点。虽然在这个例子中，BERT 学习了一些依赖项对，比如(New, city)和(York, city)，但是很明显，XLNet总是在给定相同目标的情况下学习更多的依赖项对，并且包含“更密集”的有效训练信号。


为了证明这个例子之外的一般观点，我们现在转向更正式的表达式。受以前的工作的启发,给定一个序列$\mathrm{x} = [x_1,\cdots,x_T]$,我们定义一组感兴趣的目标和上下文对,$\mathcal{I}={(x,\mathcal{U})}$, $\mathcal{U}$ 是$\mathrm{x}$中形成$x$上下文的一组tokens。凭直觉,我们想要模型通过预训练的损失项$\log p(x|\mathcal{U})$学习在$\mathcal{U}$上的$x$的依赖。例如,给出上述句子,对感兴趣的$\mathcal{I}$可以实例化:

$$
\mathcal{I}=\{(x=\mathrm{York},\mathcal{U}=\{\mathrm{New}\}),(x=\mathrm{York},\mathcal{U}=\{\mathrm{city}\}),(x=\mathrm{York},\mathcal{U}=\{\mathrm{New,city}\}),\cdots \}
$$

请注意，$\mathcal{I}$只是一个虚拟的概念，没有唯一的基本事实，我们的分析将继续，不管$\mathcal{I}$是如何实例化的。

给定一组目标 tokens $\mathcal{T}$ 和一组非目标 tokens $\mathcal{N} = \mathrm{x} \backslash \mathcal{T}$, BERT 和 XLNet 都最大化 $\log p (\mathcal{T} | \mathcal{N})$，但公式不同:

$$
\mathcal{J}_{\mathrm{BERT}} = \sum_{x\in \mathcal{T}} \log p(x|\mathcal{N});
\mathcal{J}_{\mathrm{XLNet}} = \sum_{x\in \mathcal{T}} \log p(x|\mathcal{N} \cup \mathcal{T}_{<x})
$$

其中$\mathcal{T}_{<x}$ 表示$\mathcal{T}$中因子分解顺序在$x$之前的 tokens,两个目标由多个形式如$\log p(x|\mathcal{V}_x)$的损失项组成。凭直觉，如果存在一个目标及上下文对$(x,\mathcal{U}) \in \mathcal{I}$,其中 $\mathcal{U} \subseteq \mathcal{V}_x$，那么损失项 $\log p(x|\mathcal{V}_x)$ 为$x$和$\mathcal{U}$之间的依赖提供了训练信号。为了方便起见，我们称一个目标及上下文对$(x,\mathcal{U}) \in \mathcal{I}$被模型（目标）覆盖（covered），如果 $\mathcal{U} \subseteq \mathcal{V}_x$。

根据定义，让我们考虑两种情况：
* 如果 $\mathcal{U} \subseteq \mathcal{N}$,依赖项$(x,\mathcal{U})$被 BERT 和 XLNet 覆盖。
* 如果 $\mathcal{U} \subseteq \mathcal{N} \cup \mathcal{T}_{<x}$ 且 $\mathcal{U} \cap \mathcal{T}_{<x} \ne \varnothing$，依赖关系只能由 XLNet 覆盖，不能由 BERT 覆盖。因此，XLNet 能够覆盖比 BERT 更多的依赖项。换句话说，XLNet 目标包含了更有效的训练信号，这从经验上导致了第3节中更好的性能。

### 2.6.2 与语言模型的比较

借鉴2.6.1节中的示例和符号，像 GPT 这样的标准AR语言模型只能覆盖依赖项 $(x=\mathrm{York},\mathcal{U}=\{\mathrm{New}\})$,而不能覆盖 $(x=\mathrm{New},\mathcal{U}=\{\mathrm{York}\})$。另一方面，XLNet能够满足所有因子分解顺序的期望。AR 语言建模的这种限制在实际应用中可能是至关重要的。例如，考虑一个跨度提取的问答任务，上下文为"Thom Yorke is the singer of Radiohead",问题为："Who is the singer of Radiohead"。用 AR 语言建模的话，“Thom Yorke”的表征不依赖于 "Radiohead"。因此，在所有 token 的表征上使用 softmax 的标准做法都不会选择"Thom Yorke"。更正式地说，考虑上下文及目标对 $(x,\mathcal{U})$:

* 如果 $\mathcal{U} \cap \mathcal{T}_{<x} \ne \varnothing$,其中 $\mathcal{T}_{<x}$ 表示原始序列中 $x$ 之前的 tokens，AR 语言建模不能覆盖这种依赖项。

* 相比之下，XLNet 能够覆盖预期中的所有依赖项。


像 ELMo 这样的方法以一种浅层的方式连接前向和后向的语言模型，这对于建模这两个方向之间的深度交互是不够的


### 2.6.3 弥合语言建模和预训练的差距

基于密度估计理论的语言建模是一个发展迅速的研究领域。但是，由于缺乏双向上下文建模的能力，语言建模和预训练之间存在着一定的差距，如2.6.2节所分析的。一些机器学习实践者甚至质疑，如果语言建模不能直接改善下游任务，那么它是否是一种有意义的追求。XLNet 概括了语言建模并弥补了这一缺陷。因此，它进一步“证明”了语言建模研究的合理性。此外，利用语言建模研究的快速进展进行预训练也成为可能。作为一个例子，我们将 Transformation-XL 集成到 XLNet 中，以演示最新语言建模进展的有效性。

## 3. 实验
### 3.1 预训练及实现

依照 BERT，我们使用BooksCorpus[41]和英语维基百科作为预训练数据的一部分，这些数据包含 13GB 的纯文本。此外，我们还包括 Giga5 (16GB 文本)、ClueWeb 2012-B(扩展自 Clueweb09 )和 Common Crawl 用于预训练。我们使用启发式方法积极地过滤 ClueWeb 2012-B 和 Common Crawl 中短小或低质量的文章，结果分别是 19GB 和 78GB 文本。在用 SentencePiece（arXiv preprint arXiv:1808.06226, 2018） 进行分词后，分别得到维基百科、BooksCorpus、Giga5、ClueWeb 和Common Crawl的 subword pieces 为 2.78B、1.09B、4.75B、4.30B 和 19.97B，总计32.89B。

我们最大的模型 XLNet-Large 具有与 BERT-Large 相同的体系结构超参数，这导致了相似的模型大小。序列长度和内存长度分别设置为 512 和 384。我们使用 Adam 优化器在 512 个 TPU v3芯片上对 XLNet-Large 进行了 500K 步的训练，线性学习率递减，批量大小为2048，大约需要2.5天。观察到，模型在训练结束时仍然对数据不匹配，但继续训练对下游任务没有帮助，说明在给定优化算法的情况下，模型没有足够的能力充分利用数据规模。然而，在这项工作中，我们避免训练一个更大的模型，因为它在微调的实际应用可能是有限的。此外，我们还在 BooksCorpus 和 Wikipedia 上训练一个类似于 BERT-Base 的 XLNet-Base，用于消融研究和与 BERT 的公平比较。相关结果见第3.7节。

由于引入了递归机制，我们使用了双向数据输入管道，其中每个前向和后向的方向都占用了批处理大小的一半。对于训练 XLNet-Large，我们将部分预测常数K设为6(见2.3节)。除另有规定外，我们的微调方式遵循 BERT。我们采用基于跨度（span）的预测思想，首先对长度 $L \in [1,\cdots,5]$进行采样，然后在$(KL)$的 tokens 上下文中随机选取$L$个 tokens 的连续跨度作为预测目标。

### 3.2 RACE 数据集

RACE | Accuracy | Middle | High
-|-|-|-
GPT | 59.0 | 62.9 | 57.4
BERT | 72.0 | 76.6 | 70.1
BERT+OCN∗ | 73.5 | 78.4 | 71.5
BERT+DCMN∗ | 74.1 | 79.5 | 71.8
XLNet | **81.75** | **85.45** | **80.21**

表1：，在 RACE 测试集(一个阅读理解任务)上比较最先进的结果。*表示使用了集成。“Middle”和“High”是代表中等和高等难度水平的两个子集。所有的 BERT 和 XLNet 结果都是使用24层结构得到的，具有相似的模型大小(即BERT-Large)。我们的单一模型在精确度上比最好的组合高出7.6个百分点。

RACE 数据集(arXiv preprint arXiv:1704.04683, 2017.)包含了近100K 个问题，这些问题来自12岁至18岁的中国中学生的英语考试，答案由人类专家生成。这是最困难的阅读理解数据集之一，其中包括具有挑战性的推理问题。此外， RACE 中短文的平均长度超过了300，这比其他流行的阅读理解数据集如 SQuAD 要长得多。因此，对于长文本的理解，这个数据集是一个具有挑战性的基准。在微调过程中，我们使用的序列长度为640。如表1所示，单个模型 XLNet 在精确度上比最佳集成高出7.6个百分点。同样明显的是，XLNet在很大程度上优于其他预训练的模型，如 BERT 和 GPT。由于RACE 包含相对较长的段落，我们认为 XLNet 在这个数据集上获得巨大收益的原因之一是除了 AR 目标之外，Transform-XL 体系结构的集成还提高了对长文本建模的能力。关于序列长度的更多分析见第3.7节。

### 3.3 SQuAD 数据集


没有数据增强的 dev 集结果

SQuAD1.1 |EM |F1 |SQuAD2.0 |EM |F1
-|-|-|-|-|-
BERT |84.1 |90.9 |BERT† |78.98 |81.77
XLNet |**88.95** |**94.52** |XLNet |**86.12** |**88.79**

有数据增强的排行榜上测试集结果

SQuAD1.1 |EM |F1 |SQuAD2.0 |EM |F1
-|-|-|-|-|-
Human |82.30 |91.22 |BERT+N-Gram+Self-Training |85.15 |87.72
ATB |86.94 |92.64 |SG-Net |85.23 |87.93
BERT* |87.43 |93.16 |BERT+DAE+AoA |85.88 |88.62
XLNet |**89.90** |**95.08** |XLNet |**86.35** |**89.13**


表2（上面2个表其实是一个，markdown里不能合并单元格，就给弄成2个展示了）：单独模型 XLNet 在 SQuAD1.1 上的表现比 human 和最佳集成（best ensemble) 分别高出 7.6 EM 和 2.5 EM。*表示集成（ensemble），†表示用官方代码运行。

SQuAD 是一个大型的阅读理解数据集，有两个任务。SQuAD1.1 包含的问题在给定的段落中总是有对应的答案，SQuAD2.0 则引入了无法回答的问题。为了在 SQuAD2.0 上对一个 XLNet 进行微调，我们联合应用了一个类似于分类任务的可回答性预测的逻辑回归损失和一个用于问答的标准跨度提取损失。由于 v1.1 和 v2.0 在训练集中共享相同的可回答问题，我们只需要从 v2.0 上的微调模型中删除可回答性预测部分，用于 v1.1 上的评估。由于 top 排行榜条目都采用了某种形式的数据增强，所以我们在 SQuAD2.0 和 NewsQA 上联合训练了一个 XLNet 来提升我们的排行结果。如表2所示，XLNet 在排行榜上获得了最先进的单模型结果，优于一系列基于 BERT 的方法。值得注意的是,在v1.1, XLNet单模型优于人类和最好的集成 7.6和2.5分。最后,直接与 BERT 比较，来消除在排行结果中额外的技巧的影响,我们比较 dev 集上的 XLNet 和 BERT。在 v1.1 和 v2.0 上关于 F1 值，XLNet 大大优于 BERT 3.6 和 7.0分。


### 3.4 文本分类

Model |IMDB |Yelp-2 |Yelp-5 |DBpedia |AG |Amazon-2 |Amazon-5
-|-|-|-|-|-|-|-
CNN |- |2.90 |32.39 |0.84 |6.57 |3.79 |36.24
DPCNN |- |2.64 |30.58 |0.88 |6.87 |3.32 |34.81
Mixed VAT |4.32 |- |- |0.70 |4.95 |- |-
ULMFiT |4.6 |2.16 |29.98 |0.80 |5.01 |- |-
BERT |4.51 |1.89 |29.32 |0.64 |- |2.63 |34.17
XLNet |**3.79** |**1.55** |**27.80** |**0.62** |**4.49** |**2.40** |**32.26**

表3：与几种文本分类数据集测试集的最新错误率进行比较。所有的 BERT 和 XLNet 结果都是使用24层结构得到的，具有相似的模型大小(即BERT- Large)。


依照之前的文本分类工作，我们在以下基准上评估 XLNet: IMDB、Yelp-2、Yelp-5、DBpedia、AG、Amazon-2 和 Amazon-5。从表3可以看出，XLNet 在所有考虑的数据集上都取得了新的最先进的结果，与 BERT 相比，在IMDB、Yelp-2、Yelp-5、Amazon-2 和 Amazon-5上的错误率分别降低了16%、18%、5%、9%和5%。

### 3.5 GLUE 数据集


dev 集上单独任务的单独模型
Model |MNLI |QNLI |QQP |RTE |SST-2 |MRPC |CoLA |STS-B |WNLI
-|-|-|-|-|-|-|-|-|-
BERT |86.6/- |92.3 |91.3 |70.4 |93.2 |88.0 |60.6 |90.0 |-
XLNet |**89.8/-** |**93.9** |**91.8** |**83.8** |**95.6** |**89.2** |**63.6** |**91.8** |-

test 集上单独任务的单独模型
Model |MNLI |QNLI |QQP |RTE |SST-2 |MRPC |CoLA |STS-B |WNLI
-|-|-|-|-|-|-|-|-|-
BERT |86.7/85.9 |91.1 |89.3 |70.1 |94.9 |89.3 |60.5 |87.6 |65.1
test 集上多任务集成（来自于 June 19, 2019 的排行榜）

Model |MNLI |QNLI |QQP |RTE |SST-2 |MRPC |CoLA |STS-B |WNLI
-|-|-|-|-|-|-|-|-|-
Snorkel* |87.6/87.2 |93.9 |89.9 |80.9 |96.2 |91.5 |63.8 |90.1 |65.1
ALICE* |88.2/87.9 |95.7 |**90.7** |83.5 |95.2 |92.6 |**68.6** |91.1 |80.8
MT-DNN* |87.9/87.4 |96.0 |89.9 |**86.3** |96.5 |92.7 |68.4 |91.1 |89.0
XLNet* |**90.2/89.7†** |**98.6†** |90.3† |**86.3** |**96.8†** |**93.0** |67.8 |**91.6** |**90.4**

表4（上面3个表其实是一个）：GLUE结果。*表示使用集成，†表示在多任务行里的单任务结果。所有的结果都是基于一个24层的架构，具有相似的模型大小(又称为 BERT-Large)。在公共排行榜上，最上面的行与 BERT 进行直接比较，最下面的行与最先进的结果进行比较。

GLUE 数据集是9个自然语言理解任务的集合。测试集标签将从公开发布的版本中删除，所有实践者必须在评估服务器上提交他们的预测，以获得测试集结果。在表4中，我们展示了多个设置的结果，包括单任务和多任务，以及单个模型和集成模型。在多任务设置中，我们在四个最大的数据集(MNLI、SST-2、QNLI 和 QQP)上联合训练一个 XLNet，并在其他数据集上对网络进行微调。对于四个大型数据集，只使用单任务训练。对于QNLI，我们在提交测试集时采用了《Multi-task deep neural networks
for natural language understanding. 》（arXiv preprint arXiv:1901.11504, 2019.）中的成对相关性排序方案。但是，为了与 BERT 进行公平的比较，我们对 QNLI 的 dev 集的结果是基于标准分类范例的。对于 WNLI，我们使用《A surprisingly robust trick for winograd schema challenge.》（arXiv preprint
arXiv:1905.06290, 2019.）中描述的损失。多任务集成 XLNet 在公共排行榜上的9个任务中有7个实现了最先进的结果。在基准测试范围最广的任务 MNLI 上，XLNet 将“匹配”和“不匹配”设置分别提高了 2.0 和 1.8 个百分点。请注意，排行榜上的竞争对手采用了比 BERT 更先进的技术，如蒸馏（distillation）、改进的多任务损失或元学习，但仍然低于 XLNet, XLNet除了使用标准的多任务学习方法外，没有使用其他技巧。排行榜以来,不是用于消融研究或超参调优,我们只是在测试集上评估我们最好的多任务模型。为了获得与 BERT 的直接比较,我们在 dev 集上运行一个单一任务的 XLNet。如表4上面几行所示,XLNet 始终优于 BERT,在RTE MNLI,CoLA,SST-2, STS-B上分别提高 13.4 分,3.2 分,3.0 分,2.4 分,1.8 分。


### 3.6 ClueWeb09-B 数据集


Model |NDCG@20 |ERR@20
-|-|-
DRMM |24.3 |13.8
KNRM |26.9 |14.9
Conv |28.7 |18.1
BERT† |30.53 |18.67
XLNet |**31.10** |**20.28**

表5：在ClueWeb09-B测试集上的最优结果进行比较，这是一个文档排序任务。†表示我们的实现。

按照前人工作中的设置，我们使用 ClueWeb09-B 数据集来评估文档排序的性能。这些查询是由 TREC 2009-2012 Web Tracks基于5000万个文档创建的，任务是重新排序前100个文档（使用标准检索方法检索的）。由于文档排序或临时检索主要关注低层表征，而不是高层语义，因此该数据集可作为评估单词嵌入质量的测试平台。我们使用预先训练好的 XLNet 为文档和查询提取单词嵌入，而不使用微调，并使用内核池网络(End-to-end neural
ad-hoc ranking with kernel pooling. In Proceedings of the 40th International ACM SIGIR
conference on research and development in information retrieval, pages 55–64. ACM, 2017)对文档进行排序。从表5可以看出，XLNet 大大优于其他方法，包括使用与我们相同的训练过程的 BERT 模型。这说明 XLNet 比 BERT 学到更好的底层单词嵌入。注意，为了进行公平的比较，我们排除了 Word-entity duet representations for document ranking. （In Proceedings of the 40th International ACM SIGIR conference on research and development in information retrieval, pages 763–772. ACM, 2017.）中的结果(ERR@20的结果为 19.55，比我们的略差)，因为它使用了额外的实体相关数据。

### 3.7 消融研究


|#|Model |RACE |SQuAD2.0（F1） |SQuAD2.0（EM） |MNLI(m/mm) |SST-2|
|-|-|-|-|-|-|-|
1 |BERT-Base |64.3 |76.30 |73.66 |84.34/84.65 |92.78
2 |DAE + Transformer-XL |65.03 |79.56 |76.80 |84.88/84.45 |92.60
3 |XLNet-Base (K = 7) |66.05 |**81.33** |**78.46** |**85.84/85.43** |92.66
4 |XLNet-Base (K = 6) |66.66 |80.98 |78.18 |85.63/85.12 |**93.35**
5 |- memory |65.55 |80.15 |77.27 |85.32/85.05 |92.78
6 |- span-based pred |65.95 |80.61 |77.91 |85.49/85.02 |93.12
7 |- bidirectional data |66.34 |80.65 |77.87 |85.31/84.99 |92.66
8 |+ next-sent pred |**66.76** |79.83 |76.94 |85.32/85.09 |92.89

表6：消融研究。BERT 在 RACE 上的结果来自 《matching network for multi-choice reading comprehension.》（arXiv preprint arXiv:1901.09381 2019.）。在其他数据集上我们使用官方实现运行 BERT 且与 XLNet 有相同的超参数搜索空间。K为控制优化难度的超参数(见2.3节)。所有模型都基于相同的数据进行预训练。


我们进行消融研究，以了解基于四个具有不同特征的数据集的每个设计选择的重要性。具体来说，我们希望研究的主要有三个方面:

* 对排列语言建模目标的有效性进行了研究，特别是与 BERT 使用的降噪自编码目标进行了比较。
* 使用 Transformer-XL 作为中枢神经结构和使用分段级递归(即使用 memory )的重要性。
* 需要一些实现细节，包括基于span的预测、双向输入管道和下一句话预测。

记住这些目的,在表6中,我们比较6个 XLNet-Base 的不同的实现细节的变异(3 - 8行),原 BERT-Base 模型(第1行),和一个额外 Transformer-XL 基线训练与 BERT 中使用的降噪自编码(DAE)目标,但双向输入管道(第2行)。为了公平比较,所有的模型都基于一个12层的架构，具有与 BERT-Base 相同的模型超参数，并且只在 Wikipedia 和 BooksCorpus 上训练。所有报告的结果都是5次运行的中值。

检查表6的第1 - 4行，我们看到两个采用不同$k$值训练的完整 XLNet-Base 模型在不同任务上的性能都明显优于 BERT 和 DAE 训练的 Transformer-XL，显示了排列语言建模目标的优越性。同时，有趣的是DAE 训练的 Transformer-XL 在 RACE 和 SQuAD 等长文本任务上的表现优于 BERT，说明 Transformer-XL 在语言建模方面的优势也有利于预训练。接下来一行，如果删除 memory 缓存机制(第5行)，性能将明显下降，特别是对于在4个任务中包含最长上下文的 RACE。此外，第6 - 7行表明，基于 span 的预测和双向输入管道在 XLNet 中都发挥着重要作用。最后，我们意外地发现原 BERT 中提出的下一句话预测目标并不一定会导致我们的设置有所改进。相反，除了 RACE 数据集之外，它往往会损害性能。因此，当我们训练 XLNet-Large 时，我们排除了下一句话的预测目标。

## 4. 总结

XLNet 是一种广义的 AR 预训练方法，它利用排列语言建模的目标，结合 AR 方法和 AE 方法的优点。XLNet 的神经结构被开发成与 AR 目标无缝工作，包括集成 Transformer-XL 和精心设计的两流注意力机制。XLNet 实现了最先进的结果，大大改进了各种任务。在未来，我们将 XLNet 应用于更广泛的任务，如视觉和强化学习。


## 附录 A 一种通过双流自注意力实现的目标感知（Target-Aware）表征

### A.1 一个标准LM参数化失败的具体例子

在本节中，我们提供了一个具体的例子来说明标准语言模型参数化是如何在排列目标下失败的，如2.3节所讨论的。具体来说，让我们考虑满足以下关系的两种不同排列$\mathrm{z}^{(1)}$和$\mathrm{z}^{(2)}$：

$$
\mathrm{z}_{<t}^{(1)} = \mathrm{z}_{<t}^{(2)} = \mathrm{z}_{<t}
$$
但

$$
\mathrm{z}_{t}^{(1)}=i \ne j=\mathrm{z}_{t}^{(2)}
$$

然后，将这两种排列分别代入朴素参数化，得到

$$
\underbrace{p_\theta(X_i=x|\mathrm{x}_{\mathrm{z}<t})}_{z_t^{(1)}=i,\mathrm{z}_{<t}^{(1)}=\mathrm{z}_{<t}} = 

\underbrace{p_\theta(X_j=x|\mathrm{x}_{\mathrm{z}<t})}_{z_t^{(1)}=j,\mathrm{z}_{<t}^{(1)}=\mathrm{z}_{<t}} = 

\frac{\exp(e(x)^\top h(\mathrm{x}_{\mathrm{z}<t}))}{\sum_{x'} \exp(e(x')^\top h(\mathrm{x}_{\mathrm{z}<t}))}
$$


有效地，两个不同的目标位置$i$和$j$共享完全相同的模型预测。然而，两种立场的基础事实分布肯定是不同的。


### A.2 双流注意力

在这里，我们提供了 Transformer-XL 主干的双流注意力的实现细节

初始化表征：

$$
\forall t=1,\cdots,T: \quad h_t = e(x_t) \; \mathrm{and} \; g_t = w
$$

从前一个分段中缓存 $m$ 层 content 表征（memory):$\tilde{h}^{m}$

对于 Transformer-XL 的 $m = 1,\cdots,M$ 层连续使用相对位置编码注意力和按位（position-wise）前馈更新表征:

$$
\begin{aligned}
\forall t=1,\cdots,T: \quad \hat{h}_{z_t}^{(m)} &=\mathrm{LayerNorm}(h_{z_t}^{(m-1)} + \mathrm{RelAttn(h_{z_t}^{(m-1)},[\tilde{h}^{m-1},h_{\mathrm{z}_{\le t}}^{(m-1)}])}) \\\\
h_{z_t}^{(m)} &= \mathrm{LayerNorm}(\hat{h}_{z_t}^{(m)} +\mathrm{PosFF}(\hat{h}_{z_t}^{(m)} )) \\\\

\hat{g}_{z_t}^{(m)} &=\mathrm{LayerNorm}(g_{z_t}^{(m-1)} + \mathrm{RelAttn(g_{z_t}^{(m-1)},[\tilde{h}^{m-1},h_{\mathrm{z}_{\le t}}^{(m-1)}])}) \\\\
g_{z_t}^{(m)} &= \mathrm{LayerNorm}(\hat{g}_{z_t}^{(m)} +\mathrm{PosFF}(\hat{g}_{z_t}^{(m)} ))

\end{aligned}
$$

目标感知预测的分布：

$$
p_\theta(X_{z_t}=x|\mathrm{x}_{z<t})=\frac{\exp(e(x)^\top g_{z_t}^{(M)})}{\sum_{x'}\exp(e(x')^\top g_{z_t}^{(M)})}
$$


### A.3 超参数

#### A.3.1 预训练超参数

用于 XLNet 预训练的超参数如表7所示。

Hparam |Value
-|-
Number of layers |24
Hidden size |1024
Number of attention heads |16
Attention head size |64
FFN inner hidden size |4096
Dropout |0.1
Attention dropout |0.1
Partial prediction K |6
Max sequence length |512
Memory length |384
Batch size |2048
Learning rate |1e-5
Number of steps |500K
Warmup steps |20,000
Learning rate decay |linear
Adam epsilon |1e-6
Weight decay |0.01

表7: 预训练超参数


#### A.3.2 微调超参数

用于在各种任务上对 XLNet 进行微调的超参数如表8所示。按层衰减（Layer-wise decay）意味着以自上而下的方式指数级衰减单个层的学习率。例如,假设24层的结构采用学习速率$l$,然后按层衰变率是$\alpha$,那么第$m$层的学习率是$l\alpha^{24-m}$。

Hparam | RACE | SQuAD | MNLI |Yelp-5
-|-|-|-|-
Dropout |0.1|0.1|0.1|0.1
Attention dropout |0.1|0.1|0.1|0.1
Max sequence length |512 |512 |128 |512
Batch size |32 |48 |128 |128
Learning rate |2e-5 |3e-5 |3e-5 |2e-5
Number of steps |12K |8K |10K |10K
Learning rate decay |linear|linear|linear|linear
Weight decay |0.00|0.00|0.00|0.00
Adam epsilon |1e-6 |1e-6 |1e-6 |1e-6
Layer-wise lr decay |1.0 |0.75 |1.0 |1.0

表8：微调阶段超参数

### A.4 可视化 memory 和排列

在本节中，我们将详细地展示所提出的排列语言建模目标，包括重用 memory 的机制(又称为递归机制)、如何使用注意力遮蔽来排列因子分解顺序，以及两种注意流之间的差异。如图3和图4所示，给定当前位置$z_t$，注意掩码由$\mathrm{z}$的排列(或分解顺序)决定，只有排列在$z_t$之前的才可以被注意;即位置$z_i$,其中$i<t$。此外，通过比较图3和图4，我们可以看到 query 流和 content 流如何通过注意力掩码的特定排列以不同的方式工作。主要的区别是，query 流不能进行自我注意，并且不能访问该位置的 token，而 content 流执行正常的自我注意。

![content 流](/assets/images/xlnet/内容流.jpg)

图3:长度为4的序列在因子分解顺序[3,2,4,1]下的联合视图和拆分视图，详细说明了所提议的目标的内容流。注意，如果忽略查询表征，该图中的计算只是标准的自注意力，尽管带有特定的注意力掩码

![query 流](/assets/images/xlnet/查询流.jpg)

图4:长度为4的序列在因子分解顺序[3,2,4,1]下的联合视图和拆分视图，详细说明了所提议的目标的查询流。虚线箭头表示查询流不能访问相同位置的 token (内容)，只能访问位置信息。


---
**参考**：
1. 论文：Zhilin Yang, Zihang Dai, Yiming Yang 等 [XLNet: Generalized Autoregressive Pretraining for Language Understanding](https://arxiv.org/abs/1906.08237)
2. 官方 github 代码：[xlnet](https://github.com/zihangdai/xlnet)
3. 张俊林 知乎：[XLNet:运行机制及和Bert的异同比较](https://zhuanlan.zhihu.com/p/70257427)
4. 李如 知乎：[【NLP】XLNet详解](https://zhuanlan.zhihu.com/p/70218096)