# Universal Transformers

## 简介
Transformer解决了RNN最大的缺陷：天生的序列计算结构使得它不仅无法并行处理输入序列，还具有梯度消失问题(即最大长度依赖问题)。然而值得注意的是，Transformer放弃了RNN的对学习迭代(learning iterative)和递归转换(recursive transformations)的归纳偏置（inductive bias），我们的实验表明它们在某些任务中至关重要。相比于RNN，Transformer 在更小、更加结构化的语言理解任务或简单的算法任务中（如拷贝一个字符串（如将输入「abc」转换为「abcabc」））表现欠佳，而且不是图灵完备的。

本文介绍的 Universal Transformer （UT）是一个时间并行循环自注意力序列模型（parallel-in-time recurrent self-attentive sequence model）可以作为 Transformer 模型的一种推广（generalization）。在一系列具有挑战性的序列到序列任务中提高理论能力和改进结果。UT 将 Transformer 等前馈序列模型的并行性和全局接受域与RNNs的周期性归纳偏好（recurrent inductive bias）相结合，似乎更适合一系列算法和自然语言理解的序列到序列问题。顾名思义，Universal Transformer 与标准 Transformer 的差异是，在某些假设下，UT 是图灵完备的（Turing-complete or computationally universal）


> 标准 Transformer 深度是固定的（6层）而 UT 所加的循环其实不是时间上的循环，而是depth上的循环。不同序列位置可以指定不同循环次数

![Universal Transformer](/assets/images/universal-transformer/ut_times.png)

UT 通过自注意力和循环转换函数合并不同位置的信息，在所有时间步 $1 \le t \le T$ 中多次改进并行序列中每个位置的一系列向量表征。图中用两个循环时间步来展示这个过程，箭头表示操作之间的依赖关系，$h^0$ 是用序列中每个符号的 embedding 初始化的。$h_i^t$ 表示第 $t$ 循环时间步输入符号 $i（1\le i \le m）$ 的表征。UT在序列的每一个位置也添加了动态停止机制，动态停止时，每个位置的$T$可动态确定。

实验结果表明，UT在广泛的任务范围内优于transformer和LSTMs。

## 模型

UT 基于大多数序列到序列模型常用的流形的 encoder-decoder 架构。UT 的编码器和解码器分别对输入序列和输出序列的每个位置的表征采用递归神经网络进行操作。这与大多数递归神经网络对序列数据的操作不同，UT 不会在序列中沿位置递归，而是在每个位置的向量表征（即：深度）上递归。换句话说，UT在计算上不受序列中符号数量的限制，而只受对每个符号表征的修改次数的限制。

在每个循环时间步,用两个子步骤对每个位置的表征并行处理:首先,使用self-attention机制来交换序列中所有位置的信息,从而为每个位置生成一个向量表征。然后，通过将一个转换函数(跨位置和时间共享)应用于自我注意机制的输出，每个位置的输出是独立的。由于循环转换函数可以应用任意次数，这意味着UT 可以有可变的深度(每个符号处理步骤的数量)。关键的是，这与大多数流行的神经序列模型形成了对比，包括Transformer 或 deep RNNs，它们由于应用了固定的层堆栈而具有恒定的深度。我们现在更详细地描述编码器和解码器。

**编码器**

给出长度为 $m$ 的输入序列，用$d$维的词嵌入矩阵得到序列的矩阵 $H^0 \in \mathbb{R}^{m\times d}$。UT用多头点积自注意力机制对 $m$ 个位置并行迭代计算第 $t$ 步的表征$H^t$，然后是一个循环转换函数。我们在每个功能块添加残差链接，并应用 dropout 和层归一化。

具体地说，我们使用缩放点积注意力，结合了 queries(Q),keys(K)和 values(V)

$$
Attention(Q,K,V) = Softmax(\frac{QK^T}{\sqrt{d}})V
$$

其中 $d$ 是Q，K，V的维度，用 $k$ 头的多头版本如下：

$$
MultiHeadSelfAttention(H^t) = Concat(head_1,\cdots,head_k)W^O
$$

其中

$$
head_i=Attention(H^tW_i^Q,H^tW_i^K,H^tW_i^V)
$$

我们把状态 $H^t$ 用仿射变换映射到 queries,keys,values。仿射变换的参数矩阵是可学习的，$W^Q \in \mathbb{R}^{d\times d/k}$,
$W^K \in \mathbb{R}^{d\times d/k}$,$W^V \in \mathbb{R}^{d\times d/k}$,$W^O \in \mathbb{R}^{d\times d}$

然后计算全部 $m$ 个位置的改进表征 $H^t \in \mathbb{R}^{m\times d}$

$$
H^t = LayerNorm(A^t + Transition(A^t))
$$

其中 

$$
A^t = LayerNorm((H^{t-1}+P^t)+MultiHeadSelfAttention(H^{t-1}+P^t))
$$

$LyaerNorm()$ 是Ba 等人（2016）中的层归一化，$Transition()$ 和 $P^t$ 后面讨论。

根据任务的不同，我们可以使用两个不同的转换函数中的一个：可分离卷积（Chollet,2016) 或 两个仿射变换之间的加个整流线性激活函数（rectified-linear activation function）组成的全连接神经网络，按position操作，也就是单独作用于 $A^t$ 的每一行。

上面的 $P^t \in \mathbb{R}^{m\times d}$ 是固定常量的二维（position,time）坐标嵌入。在位置 $1\le i \le m$ ，时间步 $1\le t \le T$ 的每个向量维度 $1\le j \le d$。

$$
P_{i,2j}^t = \sin(i/10000^{2j/d}) + \sin(t/10000^{2j/d}) \\\\
P_{i,2j+1}^t = \cos(i/10000^{2j/d}) + \cos(t/10000^{2j/d}) 
$$


![UT 简略结构图](/assets/images/universal-transformer/ut.png)

该图省略了位置和时间步长编码，以及dropout、残差连接和层归一化。


![UT 完整结构图](/assets/images/universal-transformer/ut_完整.png)

拥有位置和时间步嵌入，以及dropout和层归一化的完整结构图


执行 $T$ 步后（每次并行更新输入序列的所有位置），得到 UT 编码的 输入序列的 $m$ 个符号，$d$ 维的矩阵 $H^T \in \mathbb{R}^{m\times d}$


**解码器**

解码器具有与编码器相同的基本循环结构，然而，在自注意力函数后，解码器另外添加了多头点积注意力函数，用解码器表征生成 Q，用编码器最后的表征$H^T$ 生成 K 和 V。

与 Transformer 模型类似，UT 也是自回归的，用 teacher-forcing 训练。在生成时，一次生成一个符号，解码器使用先前生成的输出位置。在训练过程中，解码器的输入是输出右移一个位置。解码器的自注意力分布是未来遮蔽的，以便模型只能注意到预测符号左边的位置。最后，对解码器最终状态应用仿射变换$O \in \mathbb{R}^{d\times V}$得到每个符号的目标分布，再加上softmax,得到维度为 $(m\times V)$ 的归一化输出矩阵：

$$
p(y_{pos}|y_{[1:pos-1]},H^T) = Softmax(OH^T)

$$

为了模型生成，需要对条件输入序列运行一次编码。然后重复运行解码器，消耗所有已经生成的符号，同时在每次迭代的下一个输出位置为符号的词汇表生成一个额外的分布。然后我们通常抽样或选择概率最高的符号作为下一个符号

## 动态停止

在序列处理系统中，某些符号(如某些单词或音素)通常比其他符号更模糊。因此，将更多的处理资源分配给这些更模糊的符号是合理的。自适应计算时间(Adaptive Computation Time,ACT) (Graves, 2016)是一种在标准递归神经网络中，基于模型预测的每一步标量停止概率，动态调节处理每个输入符号所需的计算步骤数(称为“ponder time”)的机制。

受在序列中所有位置并行应用自关注 RNN 的启发，我们还为每个位置应用动态 ACT 停止机制。一旦一个符号循环块停止，它的状态将被简单地复制到下一个步骤，直到所有块停止，或者达到最大步骤数。编码器的最终输出是这样产生的最后一层的表征。

加入ACT机制的Universal transformer被称为Adaptive universal transformer.

## 总结

Universal Transformer对transformer的缺点进行了改进，1：引入了循环机制；2：在每个时间步都引入了坐标嵌入；3：自适应计算时间（ACT)。在问答、语言模型、翻译等任务上都有更好的效果，成为了新的seq2seq state-of-the-art模型。它的关键特性主要有两点：

**Weight sharing**：关于目标函数的必要假设成为归纳偏置，CNN和RNN分别假设空间转换不变性（spatial translation invariace） 和时间转换不变性（time translation invariance），体现为CNN卷积核在空间上的权重共享和RNN单元在时间上的权重共享，所以universal transformer也增加了这种假设，使recurrent机制中的权重共享，在增加了模型表达力的同时更加接近rnn的归纳偏置（inductive bias）。

**Conditional computation**：通过加入ACT控制模型的计算次数，比固定depth的universal transformer取得了更好的结果

## 代码

论文中给出了代码位置：https://github.com/tensorflow/tensor2tensor

---
**参考**：
1. 论文：Mostafa Dehghani 等人 [《Universal Transformers》](https://arxiv.org/abs/1807.03819)
2. 李如知乎：[【NLP】Universal Transformers详解](https://zhuanlan.zhihu.com/p/44655133)
3. [(简介)Universal Transformers](https://zhuanlan.zhihu.com/p/51535565)
4. [学界 | 谷歌的机器翻译模型 Transformer，现在可以用来做任何事了 ](https://www.sohu.com/a/247508556_642762)