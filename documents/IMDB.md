# IMDB

## 概述
IMDB（Internet Movie Database）是一个大型影评数据集，包含电影评论及二元情绪极性标签。他的目的是作为一个基准情绪分类。

## 数据集
包含25000个训练数据和25000个测试数据，以及50000个未标记的数据。

任何一部电影都不超过30个评论，因为同一部电影评论往往相关。另外训练集和测试集是不想交的集合，因此不能通过记住独特的电影术语和观察到的标签来得到显著的特征表现。在训练集和测试集中负面评价得分<=4/10,正面评价的得分>=7/10。因此大部分的中性评级没有包含在训练和测试集中。在非监督集中，则包含有任何的评级，>5和<=5共偶数个评论

## 下载地址
原始数据下载地址：[Large Movie Review Dataset](http://ai.stanford.edu/~amaas/data/sentiment/)


## Tensorflow中的IMDB数据集

TensorFlow中包含有IMDB数据集，并且对该数据集进行了预处理，将影评（字词序列）转换为了整数序列，其中每个整数表示字典中的一个特定字词。

以下代码会将 IMDB 数据集下载到您的计算机上（如果您已下载该数据集，则会使用缓存副本）：
```python
import tensorflow as tf
from tensorflow import keras

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)
```
参数`num_words=10000`会保留训练数据中出现频次在前 10000 位的字词。为确保数据规模处于可管理的水平，罕见字词将被舍弃。

### 将整数转换回字词
```python
# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

# The first indices are reserved
word_index = {k:(v+3) for k,v in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2  # unknown
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])
```
```python
decode_review(train_data[0])
```
```python
" this film was just brilliant casting location scenery story direction everyone's really suited the part they played and you could just imagine being there robert  is an amazing actor and now the same being director  father came from the same scottish island as myself so i loved the fact there was a real connection with this film the witty remarks throughout the film were great it was just brilliant so much that i bought the film as soon as it was released for  and would recommend it to everyone to watch and the fly fishing was amazing really cried at the end it was so sad and you know what they say if you cry at a film it must have been good and this definitely was also  to the two little boy's that played the  of norman and paul they were just brilliant children are often left out of the  list i think because the stars that play them all grown up are such a big profile for the whole film but these children are amazing and should be praised for what they have done don't you think the whole story was so lovely because it was true and was someone's life after all that was shared with us all"
```

详细内容见TensorFlow教程[影评文本分类](https://tensorflow.google.cn/tutorials/keras/basic_text_classification)