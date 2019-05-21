# FastText

fastText是一种Facebook AI Research在16年开源的一个文本分类器。 其特点就是fast。相对于其它文本分类模型，如SVM，Logistic Regression和neural network等模型，fastText在保持分类效果的同时，大大缩短了训练时间。

优点：
* **适合大型数据+高效的训练速度**：能够“在使用标准多核CPU的情况下10分钟内处理超过10亿个词汇”地训练模型
* **支持多语言表达**：利用其语言形态结构，fastText能够被设计用来支持包括英语、德语、西班牙语、法语以及捷克语等多种语言。
* **fastText专注于文本分类**
* **比word2vec更考虑了相似性**，比如 fastText 的词嵌入学习能够考虑 english-born 和 british-born 之间有相同的后缀，但 word2vec 却不能（具体参考[paper](https://arxiv.org/pdf/1607.04606v1.pdf)）。

## 模型框架

FastText模型输入词序列，输出类别概率，序列中的词和词组（n-gram）组成特征向量，特征向量通过线性变换映射到中间层，中间层再映射到标签。

fastText 在预测标签时使用了非线性激活函数，但在中间层不使用非线性激活函数。

fastText 模型架构和 Word2Vec 中的 CBOW 模型很类似。不同之处在于，fastText 预测标签，而 CBOW 模型预测中间词。如下图

![FastText](/assets/images/fasttext/fasttext_0.png)


对于输入是一个句子的 fastText 模型架构，$x_1,\cdots,x_N$是 N个 ngram 特征，经过嵌入和平均，形成了隐藏变量。如下图。

![FastText](/assets/images/fasttext/fasttext.jpeg)


对于N个文档，最小化负对数似然：

$$
-\frac{1}{N}\sum_{n=1}^N y_n \log (f(BAx_n))
$$

其中$x_n$是n个文档特征归一化后的值，$y_n$是标签，$A,B$是权重矩阵（为何BA的乘积不统一为一个矩阵呢？因为A的结果可在其他地方使用）

### Hierarchical softmax

在某些文本分类任务中类别很多，计算线性分类器的复杂度高。为了改善运行时间，fastText 模型使用了层次 Softmax 技巧。层次 Softmax 技巧建立在Huffman编码的基础上，对标签进行编码，能够极大地缩小模型预测目标的数量。可参考[Huffman编码](/documents/Huffman编码.md)和[word2vec](/documents/模型/word2vec.md)中的Hierarchical softmax

### N-gram features
常用的特征是词袋模型。但词袋模型不能考虑词之间的顺序，因此 fastText 还加入了 N-gram 特征。
“我 爱 她” 这句话中的词袋模型特征是 “我”，“爱”, “她”。这些特征和句子 “她 爱 我” 的特征是一样的。
如果加入 2-Ngram，第一句话的特征还有 “我-爱” 和 “爱-她”，这两句话 “我 爱 她” 和 “她 爱 我” 就能区别开来了。当然，为了提高效率，我们需要过滤掉低频的 N-gram。

## fastText 与 word2vec对比
1. 模型的输出层：  
word2vec的输出层，对应的是每一个term，计算某term的概率最大；而fasttext的输出层对应的是分类的label。不过不管输出层对应的是什么内容，起对应的vector都不会被保留和使用；
2. 模型的输入层：  
word2vec的输出层，是 context window 内的term；而fasttext 对应的整个sentence的内容，包括term，也包括 n-gram的内容；

3. h-softmax中向量的使用：  
word2vec的目的是得到词向量，该词向量 最终是在输入层得到，输出层对应的 h-softmax 也会生成一系列的向量，但最终都被抛弃，不会使用。  
fasttext则充分利用了h-softmax的分类功能，遍历分类树的所有叶节点，找到概率最大的label（一个或者N个）,使用了h-softmax中的向量

## 关于代码
* [官方代码](https://github.com/facebookresearch/fastText)
* [官方代码python](https://github.com/facebookresearch/fastText/tree/master/python)
* [官方python使用例子](https://github.com/facebookresearch/fastText/tree/master/python/doc/examples)
* [一个pytho接口的代码](https://github.com/salestock/fastText.py)
* [fastText源码分析以及使用](https://jepsonwong.github.io/2018/05/02/fastText/)

## 官方代码使用

官方python版安装
```
$ git clone https://github.com/facebookresearch/fastText.git
$ cd fastText
$ pip install .
```
可从官方教程[supervised-tutorial](https://fasttext.cc/docs/en/supervised-tutorial.html)中找到[数据下载](https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz)并分成训练和测试两个集合。

下方代码源码在[这里](/codes/fasttext_examples/fasttext_official_examples.py)

```python
from fastText import train_supervised,train_unsupervised,load_model

train_data = './cooking.train'
valid_data = './cooking.valid'

# 有监督训练 并测试
model = train_supervised(
    input=train_data,epoch=25,lr=1.0,wordNgrams=2,verbose=2,minCount=1)
print(*model.test(valid_data)) # 返回 N,P@k,R@K
# Read 0M words
# Number of words:  14543
# Number of labels: 735
# Progress: 100.0% words/sec/thread:   30815 lr:  0.000000 loss:  3.254056 ETA:   0h 0m
# 3000 0.5626666666666666 0.24333285281822115

# 用分层softmax的损失有监督训练（速度更快） 并测试，损失函数有ns,hs,softmax,ova四种
model = train_supervised(
    input=train_data,epoch=25,lr=1.0,wordNgrams=2,verbose=2,minCount=1,
    loss='hs'
)
print(*model.test(valid_data))
# Read 0M words
# Number of words:  14543
# Number of labels: 735
# Progress: 100.0% words/sec/thread:  602870 lr:  0.000000 loss:  2.256813 ETA:   0h 0m
# 3000 0.5446666666666666 0.2355485080005766

# 保存模型
model.save_model('cooking.bin')

# quantize模型，以减少模型的大小和内存占用。
model.quantize(input=train_data, qnorm=True, retrain=True, cutoff=100000)
print_results(*model.test(valid_data))
model.save_model("cooking.ftz")
# Progress: 100.0% words/sec/thread:  447976 lr:  0.000000 loss:  1.630192 ETA:   0h 0m
# 3000 0.5313333333333333 0.22978232665417328


# 预测
model = load_model("./cooking.bin")
print(model.predict("Which baking dish is best to bake a banana bread ?"))
print(model.predict("Why not put knives in the dishwasher?"))
# (('__label__baking',), array([0.35784602]))
# (('__label__equipment',), array([0.39477548]))

model = load_model("./cooking.ftz")
print(model.predict("Which baking dish is best to bake a banana bread ?"))
print(model.predict("Why not put knives in the dishwasher?"))
# (('__label__bread',), array([0.32475984]))
# (('__label__equipment',), array([0.49320737]))

# 无监督学习
model = train_unsupervised(input=train_data,model='skipgram')
model.save_model("cooking_uns.bin")
# Read 0M words
# Number of words:  2408
# Number of labels: 735
# Progress: 100.0% words/sec/thread:   82933 lr:  0.000000 loss:  2.764836 ETA:   0h 0m

# 查看词向量
model = load_model("cooking_uns.bin")
print("banana:",model.get_word_vector("banana"))
print("apple:",model.get_word_vector("apple"))
# 数据量太大，格式如下
# banana: [-1.87938347e-01 -4.34164740e-02  1.01463743e-01 -9.05684754e-02 ...]
# apple: [-1.83095217e-01 -4.92684692e-02  1.06943615e-01 -8.55036154e-02 ...]

# 查看label词频
model = load_model("cooking.bin")
words, freq = model.get_labels(include_freq=True)
for w, f in zip(words, freq):
    print(w + "\t" + str(f))
# 数据量太大，截取部分显示
# __label__baking 1156
# __label__food-safety    967
# __label__substitutions  724
# __label__equipment      666
# ...........

# 查看词频
model = load_model("cooking_uns.bin")
words, freq = model.get_words(include_freq=True)
for w, f in zip(words, freq):
    print(w + "\t" + str(f))
# # 数据量太大，截取部分显示
# # significant     9
# # cookers 9
# # peel?   9
# # juicy   9
# # making? 9
```

## gensim 中 FastText 的使用

```python
from gensim.models import FastText

# 训练
sentences = [["你", "是", "谁"], ["我", "是", "中国人"]]
# 方法一（官方不建议这样用）
# model = FastText(sentences,  size=4, window=3, min_count=1, iter=10,min_n = 3 , max_n = 6,word_ngrams = 1)
# 方法二
model = FastText(size=4, window=3, min_count=1, word_ngrams=1)
model.build_vocab(sentences=sentences)
model.train(sentences=sentences, total_examples=len(sentences), epochs=10)  

# 获取词向量
print(model.wv['你']) # 词向量获得的方式
print(model.wv.word_vec('你'))

# 保存模型
model.save('./model.bin')
# 加载模型
model = FastText.load("./model.bin")

# 保存词向量
model.wv.save_word2vec_format("./wv.txt")
```
结果：
```
[-0.01312488  0.01515004 -0.1250789   0.00947084]
[-0.01312488  0.01515004 -0.1250789   0.00947084]
```
---
**参考**： 
1. 论文：[Bag of Tricks for Efficient Text Classification](https://arxiv.org/abs/1607.01759)
2. 论文：[Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606)
1. [NLP︱高级词向量表达（二）——FastText（简述、学习笔记）](https://blog.csdn.net/sinat_26917383/article/details/54850933)

2. [极简使用︱Gemsim-FastText 词向量训练以及OOV（out-of-word）问题有效解决](https://blog.csdn.net/sinat_26917383/article/details/83041424)
3. [FastText原理总结](https://blog.csdn.net/qq_16633405/article/details/80578431)
1. [fasttext](https://blog.csdn.net/phoeny0201/article/details/52329477)