# BERT

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，因为decoder是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。

## 结构

![BERT GPT ELMo 对比图](/assets/images/bert/bert_gpt_elmo.jpg)
<center>BERT GPT ELMo 对比图</center>

相比于 GPT 采用的单向的 transformer ，BERT 采用的是双向 transformer。而 bert 的双向又和 ELMo 的双向有所不同，ELMo 虽然采用的双向LSTM，但其实两个单向的融合，每次只利用了单侧的语境，即分别以$P(w_i|w_1,\cdots,w_{i-1})$ 和 $P(w_i|w_{i+1},\cdots,w_n)$ 为目标函数，然后将表征拼接，而 BERT 则融合了双侧的上下文语境，作者称这为深度双向（Deep Bidirectional），即以$P(w_i|w_1,\cdots,w_{i-1},w_{i+1},\cdots,w_n)$为目标函数。

## 输入

bert输入可以是单个文本句子，也可以是一对文本句子。论文中bert要做预测下句任务，输入的是一对句子，如下图：

![Embedding](/assets/images/bert/embedding.jpg)
<center>Embedding</center>

bert的embedding有三部分组成：
* Token Embeddings 是词向量，第一个词是CLS标志，对应的输出代表整个句子的表征，如果输入两种句子，则需要用SEP表征分隔开（代码实现中句尾也有SEP标志）
* Segment Embeddings 用来区分两种句子，因为预训练不只做遮蔽语言模型（MLM）还做两个句子为输入的分类任务。代码中第一句用0标识，第二句用1标识
* Position Embeddings 和 Transformer 不太一样，不再是固定的三角函数，而是一个可学习的矩阵

## 预训练任务1：Masked LM

如果预训练模型用于处理其他任务，人们想要的肯定不止某个词左边的信息，而是左右两边的信息。ELMo 只是left-to-right 和 right-to-left 分别训练拼接起来。而作者想要做到 deeply bidirectional。普通的 LM 如果做成多层，然后预测下一个单词，那么第二层往上其实从第一层的结果中看到了要预测的单词，本来要预测的，但自己其实已经提前看到了自己（这也是为什么传统的语言模型难以做成多层rnn的原因）。而bert中采用遮蔽语言模型，虽然是多层双向，但因为要预测的单词被遮蔽了，所以并不能提前知道结果。

这里在做mask时有个track。作者随机选择15%的词汇替换为[MASK]，要求模型正确预测被替换的词，但模型真正用时是没有[MASK]标志的，这样会让模型在训练时认为输出是针对[MASK]这个标记的，但实际使用又看不到这个标记，这自然会有问题。为了避免这个问题，Bert实际上并没有完全替换这15%的词汇，而是将其中80%真正替换为 [MASK] 标记，10%不变，剩下10%随机替换成另一个单词。这就是bert的 Masked LM 模型的具体做法。


## 预训练任务2：Next Sentence Prediction

在做遮蔽语言模型任务之外，bert还同时做了句子关系预测。输入的是两个句子，这两个句子要么是语料中真正顺序相连的句子，要么第二句是随机选择拼接到第一句之后的，Next Sentence Prediction 任务就是判断第二个句子是不是真的是第一个句子的后续句子。之所以这么做，是考虑到很多NLP任务是句子关系判断任务，单词预测粒度的训练到不了句子关系这个层级，增加这个任务有助于下游句子关系判断任务。所以可以看到，它的预训练是个多任务过程。这也是Bert的一个创新。

作者特意说了语料的选取很关键，要选用document-level的而不是sentence-level的，这样可以具备抽象连续长序列特征的能力。

## 预训练阶段参数

1. 256个句子作为一个batch,每个句子最多512个token。
2. 迭代100万步。
3. 总共训练样本超过33亿。
4. 迭代40个epochs。
5. 用adam学习率， 1 = 0.9, 2 = 0.999。
6. 学习率头一万步保持固定值，之后线性衰减。
7. L2衰减，衰减参数为0.01。
8. drop out设置为0.1。
9. 激活函数用GELU代替RELU。
10. Bert base版本用了16个TPU，Bert large版本用了64个TPU，训练时间4天完成。

论文定义了两个版本:

1. base版本
* L=12
* H=768
* A=12
* Total Parameters=110M 
 
2. large版本
* L=24
* H=1024
* A=16
* Total Parameters=340M

L代表网络层数，H代表隐藏层数，A代表self attention head的数量。

## 微调

我们使用大量无标注的语料即可进行 bert 的预训练。预训练之后，即可针对具体任务进行微调。

![fine-tuning](/assets/images/bert/fine-tuning.jpg)

可以调整的参数和取值范围有：

* Batch size: 16, 32
* Learning rate (Adam): 5e-5, 3e-5, 2e-5
* Number of epochs: 3, 4

因为大部分参数都和预训练时一样，精调会快一些，所以作者推荐多试一些参数。

## 优点

BERT是截至2018年10月的最新state of the art模型，通过预训练和精调横扫了11项NLP任务，这首先就是最大的优点了。而且它还用的是Transformer，也就是相对rnn更加高效、能捕捉更长距离的依赖。对比起之前的预训练模型，它捕捉到的是真正意义上的bidirectional context信息。

## 缺点
作者在文中主要提到的就是MLM预训练时的mask问题：

1. [MASK]标记在实际预测中不会出现，训练时用过多[MASK]影响模型表现
2. 每个batch只有15%的token被预测，所以BERT收敛得比left-to-right模型要慢（它们会预测每个token）
3. 用的transformer理论上是非图灵完备的（无while功能）

---
**参考**：
1. 论文：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. 谷歌官方代码：[bert](https://github.com/google-research/bert)
3. pytorch版本代码：[pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
4. 李如 知乎：[【NLP】Google BERT详解](https://zhuanlan.zhihu.com/p/46652512)
5. 张俊林 知乎 ：[从Word Embedding到Bert模型—自然语言处理中的预训练技术发展史](https://zhuanlan.zhihu.com/p/49271699)
6. [NLP突破性成果 BERT 模型详细解读](https://zhuanlan.zhihu.com/p/46997268)