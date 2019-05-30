# TextCNN

TextCNN通过一维卷积来获取句子中n-gram的特征表示。TextCNN对文本浅层特征的抽取能力很强，在短文本领域如搜索、对话领域专注于意图分类时效果很好，应用广泛，且速度快，一般是首选；对长文本领域，TextCNN主要靠filter窗口抽取特征，在长距离建模方面能力受限，且对语序不敏感。

## TextCNN 结构
论文[《Convolutional Neural Networks for Sentence Classification》](https://arxiv.org/abs/1408.5882)中模型示意图

![论文1中的图](/assets/images/textcnn/textcnn_1.png)

论文[《A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification》](https://arxiv.org/abs/1510.03820)中模型示意图（TextCNN就看这个图吧）

![论文2中的图](/assets/images/textcnn/textcnn_2.png)

## 嵌入层

textcnn使用预先训练好的词向量作嵌入层，这个词向量矩阵可以是静态的（static），固定不变。也可以是非静态（non-static）的，可反向传播进行更新。

### CNN-rand

作为一个基础模型，Embedding layer所有words被随机初始化，然后模型整体进行训练。

### CNN-static

模型使用预训练的word2vec初始化Embedding layer，对于那些在预训练的word2vec没有的单词，随机初始化。然后固定Embedding layer，fine-tune整个网络。

### CNN-non-static

同 CNN-static，只是训练的时候，Embedding layer跟随整个网络一起训练。

### CNN-multichannel

Embedding layer有两组词向量矩阵，作为两个channel对待，一个channel为static，一个为non-static。然后整个网络fine-tune时只有一个channel更新参数。两个channel都是使用预训练的word2vec初始化的。

## 卷积

TextCNN网络中使用多个不同窗口大小的卷积核，常用的有3，4，5 filter size的卷积核。

## 池化

不同尺寸的卷积核得到的特征（feature map）大小也是不一样的，通过池化使他们维度相同，常用的就是1-max pooling,得到每个 feature map 的最大值，再几连起来得到最终的特征向量。

![MaxPooling-Over-Time](/assets/images/textcnn/MaxPooling-Over-Time.jpeg)
<center>MaxPooling-Over-Time</center>

### Max Pooling 优点：
1. 可以保证特征位置不变性，不管强特征在什么位置，都能提取出来
2. 减少模型参数，有利于减少模型过拟合
3. 把变成输入转变为固定长度

### Max Pooling 缺点：
1. 特征位置信息丢失
2. 有时候有些强特征会出现多次，出现次数越多说明这个特征越强，但是因为Max Pooling只保留一个最大值，就是说同一特征的强度信息丢失了。

### 改进

1. K-MaxPooling:  
取所有特征值中得分在Top –K的值，并保留这些特征值原始的先后顺序，即多保留一些特征信息供后续阶段使用。比如：在情感分析场景中，“我觉得这个地方景色还不错，但是人也实在太多了”，这句话前半部分表达的情感是正向的，后半部分表达的情感是负向的，显然保留Top-K最大信息能够很好的捕获这类信息。

![K-MaxPooling](/assets/images/textcnn/K-MaxPooling.jpeg)
<center>K-MaxPooling</center>

2. Chunk-MaxPooling:  
把某个Filter对应的Convolution层的所有特征向量进行分段，切割成若干段后，在每个分段里面各自取得一个最大特征值。先划分Chunk再分别取Max值的，所以保留了比较粗粒度的模糊的位置信息；当然，如果多次出现强特征，则也可以捕获特征强度。

![Chunk-MaxPooling](/assets/images/textcnn/Chunk-MaxPooling.jpeg)
<center>Chunk-MaxPooling</center>

划分方式：
1). 静态划分，设定好Chunk数再划分
2). 动态划分，根据输入不同，动态确定Chunk的边界位置，称为Chunk-Max方法。


![动态-Chunk-MaxPooling](/assets/images/textcnn/Chunk-MaxPooling.jpeg)
<center>动态-Chunk-MaxPooling</center>

关于动态Chunk-MaxPooling可参考论文：
[《Event Extraction via Dynamic Multi-Pooling Convolutional Neural
Networks》](http://www.nlpr.ia.ac.cn/cip/yubochen/yubochenPageFile/acl2015chen.pdf)

K-Max Pooling是一种全局取Top K特征的操作方式，而Chunk-Max Pooling则是先分段，在分段内包含特征数据里面取最大值，所以其实是一种局部Top K的特征抽取方式。


## 代码实现

查看[TextCNN](/codes/textcnn)

---
**参考**：
1. 论文：[Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882)
2. 论文：[A Sensitivity Analysis of (and Practitioners' Guide to) Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1510.03820)
3. [《Event Extraction via Dynamic Multi-Pooling Convolutional Neural
Networks》](http://www.nlpr.ia.ac.cn/cip/yubochen/yubochenPageFile/acl2015chen.pdf)
4. 论文翻译:[《Event Extraction via Dynamic Multi-Pooling Convolutional Neural
Networks》](https://blog.csdn.net/muumian123/article/details/82258819)
1. [深度学习：TextCNN](https://blog.csdn.net/pipisorry/article/details/85076712)