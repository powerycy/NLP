
# Transformer

## 简介
Transformer在Goole的一篇论文[Attention is All You Need](https://arxiv.org/abs/1706.03762)中被提出的。Transformer中抛弃了传统的CNN和RNN，整个网络结构完全是由Attention机制组成。更准确地讲，Transformer由且仅由self-Attenion和Feed Forward Neural Network组成。一个基于Transformer的可训练的神经网络可以通过堆叠Transformer的形式进行搭建。

作者采用Attention机制的原因是避免了RNN（或者LSTM，GRU等）依时序计算带来的以下缺点：

1. 时间片$t$的计算依赖$t-1$时刻的计算结果，这样限制了模型的并行能力；
2. 依时序计算的过程中会丢失之前较远时刻的信息，尽管LSTM等门机制的结构一定程度上缓解了长期依赖的问题，但是对于特别长期的依赖现象,LSTM依旧无能为力。

Transformer中的Attention机制，将序列中的任意两个位置之间的距离是缩小为一个常量；其次它不是类似RNN的顺序结构，因此具有更好的并行性，符合现有的GPU框架。

## 1. Transformer详解

### Transformer 主体框架
在机器翻译中，Transformer作为模型，接收一种语言的句子作为输入，将其翻译为其他语言输出。如图1：

![the_transformer_3](/assets/images/transformer/the_transformer_3.png)
<center>图1：Transformer用于机器翻译</center>

Transformer本质上是一个Encoder-Decoder的结构。如图2：

![The_transformer_encoders_decoders](/assets/images/transformer/The_transformer_encoders_decoders.png)
<center>图2：Transformer的Encoder-Decoder结构</center>

在论文中，编码组件由6个编码器堆叠而成，解码组件也有6个解码器堆叠而成，编码组件的输出会作为解码组件的输入。如图3：


![The_transformer_encoder_decoder_stack](/assets/images/transformer/The_transformer_encoder_decoder_stack.png)
<center>图3：Transformer的Encoders和Decoders的堆叠结构</center>

每个编码器的结构都完全相同，但并不共享参数，每个编码器由两部分组成：Self-Attention层和前向网络层，如图4：

![Transformer_encoder](/assets/images/transformer/Transformer_encoder.png)
<center>图4：编码器组成结构</center>

Self-Attention层可以在编码某个词时，关注到序列中其他单词的影响，后续会讲到。

解码器同样有这些子层，但两个子层间增加了attention层，用于关注到输入句子的相关部分。其结构如图5：

![Transformer_decoder](/assets/images/transformer/transformer_decoder.png)
<center>图5：解码器组成结构</center>

### 数据流动

现在我们看看数据是如何流经各个组件并输出的。

如NLP中常见例子，先将输入单词通过词嵌入方法转化成向量,如图6。论文中使用的词嵌入的维度为$d_{model}=512$。

![embeddings](/assets/images/transformer/embeddings.png)
<center>图6：embeddings</center>

词的向量化仅仅发生在最底层的编码器的输入时，这样每个编码器的都会接收到一个list（每个元素都是512维的词向量），只不过其他编码器的输入是前个编码器的输出。list的尺寸是可以设置的超参，通常是训练集的最长句子的长度。为了画图更简单，我们使用更简单的例子来表示接下来的过程，如图7：

![encoder_with_tensors_2](/assets/images/transformer/encoder_with_tensors_2.png)
<center>图7：输入数据经embeddings后的流动</center>

这里能看到Transformer的一个关键特性，每个位置的词仅仅流过它自己的编码器路径。在self-attention层中，这些路径两两之间是相互依赖的。前向网络层则没有这些依赖性，但这些路径在流经前向网络时可以并行执行。

### Self-Attention 思路

在理解 Self-Attention 前我们先看一个例子，作为我们想要翻译的输入语句“The animal didn’t cross the street because it was too tired”。句子中"it"指的是什么呢？“it"指的是"street” 还是“animal”？对人来说很简单的问题，但是对算法而言并不简单。

当模型处理单词“it”时，self-attention允许将“it”和“animal”联系起来。当模型处理某个位置的词时，self-attention允许模型看到句子的其他位置信息作辅助线索来更好地编码当前词。如图8：

![transformer_self-attention_visualization](/assets/images/transformer/transformer_self-attention_visualization.png)
<center>图8：经典Attention可视化示例图</center>

当编码"it"时（编码器的最后层输出），部分attention集中于"the animal"，并将其表示合并进入到“it”的编码中

### Self-Attention 计算详情

1. 根据编码器的输入向量，生成三个向量。  
在Self-Attention中，每个单词向量分别乘以权重矩阵$W^Q$,$W^K$,$W^V$,生成3个向量Query向量（ Q ），Key向量（ K ）和Value向量（ V ），长度均是64。

$$
q_i = x_i W^Q \\
k_i = x_i W^K \\
v_i = x_i W^V \\
$$  

图示如图9：

![transformer_self_attention_vectors](/assets/images/transformer/transformer_self_attention_vectors.png)
<center>图9：Q，K，V的计算示例图</center>

2. 计算 attention 分值  
以“Thinking Matchines”这句话为例，对于第一个词“Thinking”，我们需要计算“Thinking”和每个词的评估分，来决定编码“Thinking”时需要对每个词的关注度。

这个分值是通过“Thinking”对一个的Query向量与所有词的Key向量依次做点积得到。

计算“Thinking”的attention分值过程如图10：

![transformer_self_attention_score](/assets/images/transformer/transformer_self_attention_score.png)
<center>图10：self_attention_score</center>

3. 除以8（=$\sqrt{dim_{key}}$）,这样梯度会更稳定。  
如果Q，K的维度过大，那么他们的内积值也将会很大，经过后面的softmax后，梯度更新时对应的梯度将会很小。为了抵消这种影响，所以需要将内积除以$\sqrt{dim_{key}}$
4. 进行softmax，归一化分值使得全为整数且和为1。
5. 将softmax分值与Value按位相乘。保留关注此的value值，削减非相关词的value值。
6. 将所有加权向量求和，产生该位置的Self-Attention的输出结果

完整的计算过程如图11：

![self-attention-output](/assets/images/transformer/self-attention-output.png)
<center>图11：self_attention_score</center>

我们可以把所有输入词向量合并成输入矩阵X，这样整个计算过程可以用矩阵形式表示，如图12，13：


![self-attention-matrix-calculation](/assets/images/transformer/self-attention-matrix-calculation.png)
<center>图12：self_attention 矩阵运算示意图1</center>

![self-attention-matrix-calculation-2](/assets/images/transformer/self-attention-matrix-calculation-2.png)
<center>图13：self_attention 矩阵运算示意图2</center>


### 多头注意力机制（Multi-Head Attention）

论文中给self-attention进一步增加了multi-headed机制，来提高attention层的效果：
1. 多头机制扩展了模型集中于不同位置的能力。如上例，z1只包含了极少的其他词的信息，绝大部分信息是由他自己决定的。而在翻译“The animal didn’t cross the street because it was too tired”时，单词“it”就需要更多的包含其他词的信息，比如“animal”。
2. 多头机制赋予attention多种子表达方式。在多头下有多组Q，K，V矩阵，而非仅仅一组（论文中使用的是8-heads),如图14。每一组都是随机初始化，经过训练之后，输入向量可以被映射到不同的子表达空间中。

![transformer_attention_heads_qkv](/assets/images/transformer/transformer_attention_heads_qkv.png)
<center>图14：多头Q,K,V</center>

如果我们计算multi-headed self-attention的，分别有八组不同的Q、K、V 矩阵，我们得到八个不同的矩阵。如图15：

![transformer_attention_heads_z](/assets/images/transformer/transformer_attention_heads_z.png)
<center>图15：多头得到多个输出结果Z</center>

但我们的前向网络希望输入的是一个矩阵，而不能接受八个矩阵。所以需要将八个矩阵合并成一个矩阵。如图16：

![transformer_attention_heads_weight_matrix_o](/assets/images/transformer/transformer_attention_heads_weight_matrix_o.png)
<center>图16：多头结果转换为一个矩阵</center>

完整过程如图17：


![transformer_multi-headed_self-attention-recap](/assets/images/transformer/transformer_multi-headed_self-attention-recap.png)
<center>图17：多头注意力完整运算示意图</center>

加入attention heads后，看看下面的例子图18：


![transformer_self-attention_visualization_2](/assets/images/transformer/transformer_self-attention_visualization_2.png)
<center>图18：多头注意力例子</center>

编码"it"时，一个attention head集中于"the animal"，另一个head集中于“tired”，某种意义上讲，模型对“it”的表达合成了的“animal”和“tired”两者

### 使用位置编码表示序列顺序

截止目前为止，我们介绍的Transformer模型并没有捕捉顺序序列的能力，也就是说无论句子的结构怎么打乱，Transformer都会得到类似的结果。换句话说，Transformer只是一个功能更强大的词袋模型而已。

为了解决这个问题，论文中在编码词向量时引入了位置编码（Position Embedding）的特征。具体地说，位置编码会在词向量中加入了单词的位置信息，这样Transformer就能区分不同位置的单词了。

那么怎么编码这个位置信息呢？常见的模式有：a. 根据数据学习；b. 自己设计编码规则。在这里作者采用了第二种方式。那么这个位置编码该是什么样子呢？通常位置编码是一个长度为$d_{model}$的特征向量，这样便于和词向量进行单位加的操作，如图19。需要注意的是，编码方法必须能够处理未知长度的序列。


![transformer_positional_encoding_vectors.png](/assets/images/transformer/transformer_positional_encoding_vectors.png)
<center>图19：Position Embedding</center>

论文给出的编码公式如下：

$$
PE(pos,2i)= \sin(\frac{pos}{10000^{\frac{2i}{d_{model}}}}) \\
PE(pos,2i+1)= \cos(\frac{pos}{10000^{\frac{2i}{d_{model}}}})
$$

在上式中，$pos$表示单词的位置，$i$表示单词的维度。关于位置编码的实现可在Google开源的算法中[get_timing_signal_1d()](https://link.zhihu.com/?target=https%3A//github.com/tensorflow/tensor2tensor/blob/23bd23b9830059fbc349381b70d9429b5c40a139/tensor2tensor/layers/common_attention.py)函数找到对应的代码。

作者这么设计的原因是考虑到在NLP任务重，除了单词的绝对位置，单词的相对位置也非常重要。根据公式$sin(\alpha+\beta) = sin \alpha cos \beta + cos \alpha sin\beta$以及$cos(\alpha + \beta) = cos \alpha cos \beta - sin \alpha sin\beta$，这表明位置$k+p$的位置向量可以表示为位置$k$的特征向量的线性变化，这为模型捕捉单词之间的相对位置关系提供了非常大的便利。

### 残差（The Residuals）

编码器结构中值得提出注意的一个细节是，在每个子层中（slef-attention, ffnn），都有残差连接，并且紧跟着[layer-normalization](https://arxiv.org/abs/1607.06450)。如图20，21


![transformer_resideual_layer_norm.png](/assets/images/transformer/transformer_resideual_layer_norm.png)
<center>图20：残差</center>


![transformer_resideual_layer_norm_2.png](/assets/images/transformer/transformer_resideual_layer_norm_2.png)
<center>图21：可视化向量的残差示意图</center>

在解码器中也是如此，假设两层编码器+两层解码器组成Transformer，其结构如图22：

![transformer_resideual_layer_norm_3.png](/assets/images/transformer/transformer_resideual_layer_norm_3.png)
<center>图22：完整的编码解码示意图</center>

### 解码器

在解码器中的self attention层与编码器中稍有不同。由于在机器翻译中，解码过程是一个顺序操作的过程，也就是当解码第$k$个特征向量时，我们只能看到第$k-1$及其之前的解码结果。在softmax之前，我们通过遮挡未来位置（将他们设置为-inf）来实现。论文中把这种情况下的multi-head attention叫做masked multi-head attention。如图23：

![the-annotated-transformer_14_0.png](/assets/images/transformer/the-annotated-transformer_14_0.png)
<center>图23：论文中的示意图</center>


解码器也比编码器多了个encoder-cecoder attention,他的工作方式与multi-head self-attention是一样的，不同的是，在encoder-cecoder attention中，query矩阵是由前层的输出转化得到，而key和value矩阵是由编码器最后的输出转化得到。

### 整个编解码过程

编码器从输入序列的处理开始，最后的编码器的输出被转换为K和V，它俩被每个解码器的"encoder-decoder atttention"层来使用，帮助解码器集中于输入序列的合适位置。如图24：


![transformer_decoding_1](/assets/images/transformer/transformer_decoding_1.gif)
<center>图24：编码过程</center>


解码的每一步输出一个元素作输出序列。一直重复直到一个特殊符号出现表示解码器完成了翻译输出。每一步的输出被喂到下一个解码器中。正如编码器的输入所做的处理，对解码器的输入增加位置向量。如图25：


![transformer_decoding_2](/assets/images/transformer/transformer_decoding_2.gif)
<center>图25：解码过程</center>

### 最后的线性变换及softmax层

解码器最后输出浮点向量，如何将它转成词？这是最后的线性层和softmax层的主要工作。

线性层是个简单的全连接层，将解码器的最后输出映射到一个非常大的logits向量上。假设模型已知有1万个单词（输出的词表）从训练集中学习得到。那么，logits向量就有1万维，每个值表示是某个词的可能倾向值。
softmax层将这些分数转换成概率值（都是正值，且加和为1），最高值对应的维上的词就是这一步的输出单词。如图26：


![transformer_decoder_output_softmax](/assets/images/transformer/transformer_decoder_output_softmax.png)
<center>图26：解码后的输出变换</center>

### 损失函数
由于模型参数是随机初始化的，未训练的模型输出随机值。我们可以对比真实输出，然后利用误差后传调整模型权重，使得输出更接近与真实输出。如何对比两个概率分布呢？简单采用 cross-entropy或者Kullback-Leibler divergence中的一种。


### 总结

优点：

1. 虽然Transformer最终也没有逃脱传统学习的套路，Transformer也只是一个全连接（或者是一维卷积）加Attention的结合体。但是其设计已经足够有创新，因为其抛弃了在NLP中最根本的RNN或者CNN并且取得了非常不错的效果，算法的设计非常精彩，值得每个深度学习的相关人员仔细研究和品位。
2. Transformer的设计最大的带来性能提升的关键是将任意两个单词的距离是1，这对解决NLP中棘手的长期依赖问题是非常有效的。
3. Transformer不仅仅可以应用在NLP的机器翻译领域，甚至可以不局限于NLP领域，是非常有科研潜力的一个方向。
4. 算法的并行性非常好，符合目前的硬件（主要指GPU）环境。

缺点：

1. 粗暴的抛弃RNN和CNN虽然非常炫技，但是它也使模型丧失了捕捉局部特征的能力，RNN + CNN + Transformer的结合可能会带来更好的效果。
2. Transformer失去的位置信息其实在NLP中非常重要，而论文中在特征向量中加入Position Embedding也只是一个权宜之计，并没有改变Transformer结构上的固有缺陷。

## 2. Transformer代码梳理
该代码实现来源于
[The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)。文章中提供了[github](https://github.com/harvardnlp/annotated-transformer)和[colab](https://drive.google.com/file/d/1xQXSv6mtAOLXxEMi8RvaW8TW-7bvYBDF/view?usp=sharing)两处代码。建议直接查看[colab](https://drive.google.com/file/d/1xQXSv6mtAOLXxEMi8RvaW8TW-7bvYBDF/view?usp=sharing)中的那个代码，最好自己能跟着写一遍，运行一下。


待写




---
**参考**：
1. 论文 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
2. [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
3. [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
4. [The Illustrated Transformer【译】](https://blog.csdn.net/yujianmin1990/article/details/85221271)
5. [图解Transformer](https://blog.csdn.net/qq_41664845/article/details/84969266)
6. [详解Transformer （Attention Is All You Need）](https://zhuanlan.zhihu.com/p/48508221)
7. [代码参考](https://drive.google.com/file/d/1xQXSv6mtAOLXxEMi8RvaW8TW-7bvYBDF/view?usp=sharing)