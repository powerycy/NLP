# BERT-wwm

## 摘要
基于Transformers的双向编码表示（BERT）在多个自然语言处理任务中取得了广泛的性能提升。近期，谷歌发布了基于全词覆盖（Whold Word Masking）的BERT预训练模型，并且在SQuAD数据中取得了更好的结果。应用该技术后，在预训练阶段，同属同一个词的WordPiece会被全部覆盖掉，而不是孤立的覆盖其中的某些WordPiece，进一步提升了Masked Language Model （MLM）的难度。在本文中我们将WWM技术应用在了中文BERT中。我们采用中文维基百科数据进行了预训练。该模型在多个自然语言处理任务中得到了测试和验证，囊括了句子级到篇章级任务，包括：情感分类，命名实体识别，句对分类，篇章分类，机器阅读理解。实验结果表明，基于全词覆盖的中文BERT能够带来进一步性能提升。同时我们对现有的中文预训练模型BERT，ERNIE和本文的BERT-wwm进行了对比，并给出了若干使用建议。预训练模型将发布在：https://github.com/ymcui/Chinese-BERT-wwm

## 简介

**Whole Word Masking (wwm)**，暂且翻译为`全词Mask`，是谷歌在2019年5月31日发布的一项BERT的升级版本，主要更改了原预训练阶段的训练样本生成策略。简单来说，原有基于WordPiece的分词方式会把一个完整的词切分成若干个词缀，在生成训练样本时，这些被分开的词缀会随机被mask。在`全词Mask`中，如果一个完整的词的部分WordPiece被mask，则同属该词的其他部分也会被mask，即`全词Mask`。

**需要注意的是，这里的mask指的是广义的mask（替换成[MASK]；保持原词汇；随机替换成另外一个词），并非只局限于单词替换成`[MASK]`标签的情况。更详细的说明及样例请参考：[issue-4](https://github.com/ymcui/Chinese-BERT-wwm/issues/4)**

同理，由于谷歌官方发布的`BERT-base , Chinese`中，中文是以**字**为粒度进行切分，没有考虑到传统NLP中的中文分词（CWS）。我们将全词Mask的方法应用在了中文中，使用了中文维基百科（包括简体和繁体）进行训练，并且使用了[哈工大LTP](http://ltp.ai)作为分词工具），即对组成同一个**词**的汉字全部进行Mask。

下述文本展示了`全词Mask`的生成样例（注意：为了方便理解，下述例子中只考虑替换成[MASK]标签的情况。）。

| 说明 | 样例 |
| :------- | :--------- |
| 原始文本 | 使用语言模型来预测下一个词的probability。 |
| 分词文本 | 使用 语言 模型 来 预测 下 一个 词 的 probability 。 |
| 原始Mask输入 | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity 。 |
| 全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |

## 模型

### 1. 数据处理

我们下载了最新的 [Wikipedia dump](https://dumps.wikimedia.org/zhwiki/latest/) 并按照Devlin 等人的建议，使用 WikiExtractor.py 进行预处理，得到1307个提取文件。注意：我们同时使用了 dump 中的简体和繁体中文。在清洗完原始数据（如移除html标签）和拆分文档后，我们得到了 13.6M 行的最终输入文本。为了识别汉语单词的边界，我们使用了 LTP6 进行了汉语分词。我们使用官方 BERT Github 仓库提供的 create_pretraining_data.py 将原始输入文本转换为预训练的样本。我们按照 Devlin 等人的建议，生成了两组最大长度分别为 128 和 512 的预训练样本，来提高计算效率和学习长期依赖。我们严格遵守原始 wwm 代码并且不改变其他部分，例如遮蔽单词的百分比等。例如下图：

![bert-wwm例子](/assets/images/bert-wwm/wwm.jpeg)
<center>bert-wwm例子</center>

### 2.预训练

我们认为全词mask是 BERT 学习单词边界的一种方法，这比起创建一个新模型，算是一种补救方法。在这种假定之下，我们并没有从头开始训练模型，而是基于官方 BERT-base(Chinese) 上训练的。我们用最大长度128，batch size 为2560，初始学习率 1e-4(warm-up ratio 10%) 训练了100k步，然后在最大长度512，batch size 为384 上训练了 100k 步来学习长期依赖和位置嵌入（position embeddings）。注意，这两个阶段的输入应该根据最大长度来改变。我们并没有使用原始BERT中的 AdamWeightDecayOptimizer,而是使用 LAMB 优化器在大批次中获得更好的可伸缩性。我们在 128G HBM 的 Google Cloud TPU v3 上做的预训练。

### 3. 下游任务上的微调

使用这个模型很简单，只需一步：用我们的模型替换原始的中文 BERT，无需改变 config 和 vocabulary 文件

## 中文模型下载
*   [**`BERT-base, Chinese (Whole Word Masking)`**](https://storage.googleapis.com/hfl-rc/chinese-bert/chinese_wwm_L-12_H-768_A-12.zip): 
    12-layer, 768-hidden, 12-heads, 110M parameters

#### TensorFlow版本（1.12、1.13、1.14测试通过）
- Google: [download_link_for_google_storage](https://storage.googleapis.com/hfl-rc/chinese-bert/chinese_wwm_L-12_H-768_A-12.zip)
- 讯飞云: [download_link_密码mva8](https://pan.iflytek.com:443/link/4B172939D5748FB1A3881772BC97A898)

#### PyTorch版本（请使用🤗 的[PyTorch-BERT](https://github.com/huggingface/pytorch-pretrained-BERT) > 0.6，其他版本请自行转换）
- Google: [download_link_for_google_storage](https://storage.googleapis.com/hfl-rc/chinese-bert/chinese_wwm_pytorch.zip)
- 讯飞云: [download_link_密码m1CE](https://pan.iflytek.com:443/link/F23B12B39A3077CF1ED7A08DDAD081E3)

中国大陆境内建议使用讯飞云下载，境外用户建议使用谷歌云下载点，文件大小约**400M**。
以TensorFlow版本为例，下载完毕后对zip文件进行解压得到：
```
chinese_wwm_L-12_H-768_A-12.zip
    |- bert_model.ckpt      # 模型权重
    |- bert_model.meta      # 模型meta信息
    |- bert_model.index     # 模型index信息
    |- bert_config.json     # 模型参数
    |- vocab.txt            # 词表
```
其中`bert_config.json`和`vocab.txt`与谷歌原版`**BERT-base, Chinese**`完全一致。

## 基线测试结果

查看 github 仓库：[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) 或 微信公众号文章介绍：[哈工大讯飞联合实验室发布基于全词覆盖的中文BERT预训练模型](https://mp.weixin.qq.com/s/EE6dEhvpKxqnVW_bBAKrnA)


## 使用建议
* 初始学习率是非常重要的一个参数（不论是BERT还是其他模型），需要根据目标任务进行调整。
* ERNIE的最佳学习率和BERT/BERT-wwm相差较大，所以使用ERNIE时请务必调整学习率（基于以上实验结果，ERNIE需要的初始学习率较高）。
* 由于BERT/BERT-wwm使用了维基百科数据进行训练，故它们对正式文本建模较好；而ERNIE使用了额外的百度百科、贴吧、知道等网络数据，它对非正式文本（例如微博等）建模有优势。
* 在长文本建模任务上，例如阅读理解、文档分类，BERT和BERT-wwm的效果较好。
* 如果目标任务的数据和预训练模型的领域相差较大，请在自己的数据集上进一步做预训练。
* 如果要处理繁体中文数据，请使用BERT或者BERT-wwm。因为我们发现ERNIE的词表中几乎没有繁体中文。


---
**参考**：

1. 论文：[Pre-Training with Whole Word Masking for Chinese BERT](https://arxiv.org/abs/1906.08101)
2. github 仓库：[Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm)
3. 微信公众号文章介绍：[哈工大讯飞联合实验室发布基于全词覆盖的中文BERT预训练模型](https://mp.weixin.qq.com/s/EE6dEhvpKxqnVW_bBAKrnA)
4. [中文最佳，哈工大讯飞联合发布全词覆盖中文BERT预训练模型](https://mbd.baidu.com/newspage/data/landingsuper?context=%7B%22nid%22%3A%22news_9942975897275092477%22%7D&n_type=0&p_from=1)