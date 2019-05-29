# HAN

## 数据
使用的是 THUCNews 的一个子集，下载地址：链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud

(查看[TextCNN](/codes/textcnn))
(查看[TextRNN](/codes/textrnn))

创建两个文件夹：data,output。将数据集中的 txt 文件放置在data目录下。output到时候用来保存训练好的模型

## 配置
`config.yaml`为配置文件，这只是简单的小例子，供学习 HAN 的原理使用的，完全没有调参。

## 运行过程

运行`predata.py`中的函数
1. 分词
```python
segment("./data/cnews.train.txt","./data/cnews_seg_train.txt")
segment("./data/cnews.test.txt","./data/cnews_seg_test.txt")
```
2. word2vec 预训练词向量
```python
w2v("./data/cnews_seg_train.txt","./data/vectors")
```
3. 保留常用的20000个词的词向量作为词典(cnn,rnn都用的5k,感觉有好多有意义的词汇没包含进去，所以改成了2w)
```python
 top_k_vec("./data/vectors","./data/stop_words.txt","./data/final_vectors",20000)
```
4. 创建标签文件
```python
label_list("./data/cnews_seg_train.txt","./data/label")
```
5. 将原始数据id化
```python
file2id("./data/cnews_seg_train.txt","./data/label","./data/final_vectors","./data/cnews_final_train.txt")
file2id("./data/cnews_seg_test.txt","./data/label","./data/final_vectors","./data/cnews_final_test.txt")
```
6. 查看语料长度分布（可运行，也可不运行）

```python
plot_sentences_distribution("./data/cnews_final_train.txt")
```
7. 执行`train.py`中的 train 函数进行训练
8. 执行`train.py`中的 test 函数进行测试


## 结果
训练过程。cnn,rnn都是超过1000批次没提升就结束了，这里改成了2000批次没提升结束，不过没多大差别。主要是觉得这个模型参数多，可能训练的久，就把epoch改大了一倍，所以等待多久没提升就结束的参数也给改大一倍。
```
# 输出Epoch:0
# Iter:    100, Train Loss:   0.34, Train Acc:  92.19%, Val Loss:   0.52, Val Acc:  85.45%, Time: 0:01:05 *
# Iter:    200, Train Loss:   0.36, Train Acc:  90.62%, Val Loss:   0.32, Val Acc:  91.53%, Time: 0:02:06 *
# Iter:    300, Train Loss:   0.19, Train Acc:  93.75%, Val Loss:   0.18, Val Acc:  94.70%, Time: 0:03:07 *
# Iter:    400, Train Loss:   0.13, Train Acc:  93.75%, Val Loss:   0.23, Val Acc:  92.94%, Time: 0:04:07 
# Iter:    500, Train Loss:   0.33, Train Acc:  87.50%, Val Loss:   0.47, Val Acc:  84.31%, Time: 0:05:08 
# Iter:    600, Train Loss:   0.14, Train Acc:  95.31%, Val Loss:   0.21, Val Acc:  93.92%, Time: 0:06:08 
# Iter:    700, Train Loss:  0.086, Train Acc:  98.44%, Val Loss:   0.14, Val Acc:  95.51%, Time: 0:07:09 *
# Epoch:1
# Iter:    800, Train Loss:   0.08, Train Acc:  96.88%, Val Loss:   0.13, Val Acc:  95.94%, Time: 0:08:09 *
# Iter:    900, Train Loss:  0.054, Train Acc:  98.44%, Val Loss:   0.12, Val Acc:  96.32%, Time: 0:09:09 *
# Iter:   1000, Train Loss:  0.096, Train Acc:  93.75%, Val Loss:   0.13, Val Acc:  96.02%, Time: 0:10:10 
# Iter:   1100, Train Loss:   0.11, Train Acc:  93.75%, Val Loss:   0.11, Val Acc:  96.81%, Time: 0:11:11 *
# Iter:   1200, Train Loss:  0.031, Train Acc: 100.00%, Val Loss:   0.13, Val Acc:  96.05%, Time: 0:12:12 
# Iter:   1300, Train Loss:  0.079, Train Acc:  96.88%, Val Loss:   0.11, Val Acc:  96.71%, Time: 0:13:12 
# Iter:   1400, Train Loss:  0.065, Train Acc:  96.88%, Val Loss:   0.13, Val Acc:  96.01%, Time: 0:14:12 
# Iter:   1500, Train Loss:  0.097, Train Acc:  95.31%, Val Loss:   0.14, Val Acc:  95.34%, Time: 0:15:13 
# Epoch:2
# Iter:   1600, Train Loss: 0.0052, Train Acc: 100.00%, Val Loss:  0.093, Val Acc:  97.41%, Time: 0:16:13 *
# Iter:   1700, Train Loss:  0.036, Train Acc:  98.44%, Val Loss:  0.091, Val Acc:  97.18%, Time: 0:17:14 
# Iter:   1800, Train Loss:  0.013, Train Acc: 100.00%, Val Loss:  0.088, Val Acc:  97.52%, Time: 0:18:14 *
# Iter:   1900, Train Loss:   0.12, Train Acc:  95.31%, Val Loss:    0.1, Val Acc:  97.22%, Time: 0:19:15 
# Iter:   2000, Train Loss:  0.017, Train Acc:  98.44%, Val Loss:   0.11, Val Acc:  96.81%, Time: 0:20:15 
# Iter:   2100, Train Loss:  0.019, Train Acc: 100.00%, Val Loss:   0.13, Val Acc:  96.49%, Time: 0:21:16 
# Iter:   2200, Train Loss:  0.082, Train Acc:  98.44%, Val Loss:    0.1, Val Acc:  96.85%, Time: 0:22:16 
# Iter:   2300, Train Loss:    0.1, Train Acc:  98.44%, Val Loss:   0.21, Val Acc:  93.36%, Time: 0:23:17 
# Epoch:3
# Iter:   2400, Train Loss:  0.034, Train Acc:  98.44%, Val Loss:   0.19, Val Acc:  94.43%, Time: 0:24:17 
# Iter:   2500, Train Loss:   0.04, Train Acc:  98.44%, Val Loss:   0.16, Val Acc:  95.44%, Time: 0:25:17 
# Iter:   2600, Train Loss:  0.082, Train Acc:  96.88%, Val Loss:   0.13, Val Acc:  96.20%, Time: 0:26:18 
# Iter:   2700, Train Loss:  0.039, Train Acc:  98.44%, Val Loss:   0.11, Val Acc:  97.18%, Time: 0:27:18 
# Iter:   2800, Train Loss:  0.045, Train Acc:  98.44%, Val Loss:  0.099, Val Acc:  97.08%, Time: 0:28:19 
# Iter:   2900, Train Loss: 0.0027, Train Acc: 100.00%, Val Loss:   0.11, Val Acc:  96.73%, Time: 0:29:19 
# Iter:   3000, Train Loss:  0.011, Train Acc: 100.00%, Val Loss:  0.087, Val Acc:  97.37%, Time: 0:30:20 
# Iter:   3100, Train Loss:  0.012, Train Acc: 100.00%, Val Loss:    0.1, Val Acc:  97.12%, Time: 0:31:20 
# Epoch:4
# Iter:   3200, Train Loss:  0.011, Train Acc: 100.00%, Val Loss:   0.08, Val Acc:  97.73%, Time: 0:32:20 *
# Iter:   3300, Train Loss:  0.035, Train Acc:  98.44%, Val Loss:   0.11, Val Acc:  97.03%, Time: 0:33:20 
# Iter:   3400, Train Loss:  0.029, Train Acc: 100.00%, Val Loss:   0.11, Val Acc:  96.97%, Time: 0:34:21 
# Iter:   3500, Train Loss:  0.011, Train Acc: 100.00%, Val Loss:  0.095, Val Acc:  97.44%, Time: 0:35:21 
# Iter:   3600, Train Loss:  0.056, Train Acc:  98.44%, Val Loss:   0.11, Val Acc:  96.78%, Time: 0:36:21 
# Iter:   3700, Train Loss:  0.018, Train Acc:  98.44%, Val Loss:   0.11, Val Acc:  96.83%, Time: 0:37:22 
# Iter:   3800, Train Loss:  0.059, Train Acc:  98.44%, Val Loss:    0.1, Val Acc:  97.11%, Time: 0:38:22 
# Iter:   3900, Train Loss:  0.016, Train Acc: 100.00%, Val Loss:  0.088, Val Acc:  97.80%, Time: 0:39:23 *
# Epoch:5
# Iter:   4000, Train Loss:  0.043, Train Acc:  98.44%, Val Loss:    0.2, Val Acc:  94.59%, Time: 0:40:23 
# Iter:   4100, Train Loss:  0.017, Train Acc: 100.00%, Val Loss:  0.079, Val Acc:  97.89%, Time: 0:41:24 *
# Iter:   4200, Train Loss: 0.0069, Train Acc: 100.00%, Val Loss:    0.1, Val Acc:  97.20%, Time: 0:42:24 
# Iter:   4300, Train Loss:  0.029, Train Acc:  98.44%, Val Loss:    0.1, Val Acc:  97.41%, Time: 0:43:25 
# Iter:   4400, Train Loss:  0.055, Train Acc:  98.44%, Val Loss:   0.14, Val Acc:  96.50%, Time: 0:44:25 
# Iter:   4500, Train Loss: 0.00066, Train Acc: 100.00%, Val Loss:   0.12, Val Acc:  97.19%, Time: 0:45:25 
# Iter:   4600, Train Loss:  0.081, Train Acc:  98.44%, Val Loss:   0.14, Val Acc:  96.76%, Time: 0:46:26 
# Epoch:6
# Iter:   4700, Train Loss:  0.014, Train Acc: 100.00%, Val Loss:  0.087, Val Acc:  97.51%, Time: 0:47:26 
# Iter:   4800, Train Loss: 0.0083, Train Acc: 100.00%, Val Loss:   0.11, Val Acc:  97.12%, Time: 0:48:26 
# Iter:   4900, Train Loss: 0.0021, Train Acc: 100.00%, Val Loss:  0.096, Val Acc:  97.31%, Time: 0:49:27 
# Iter:   5000, Train Loss: 0.0024, Train Acc: 100.00%, Val Loss:   0.12, Val Acc:  96.93%, Time: 0:50:27 
# Iter:   5100, Train Loss: 0.0095, Train Acc: 100.00%, Val Loss:  0.082, Val Acc:  97.83%, Time: 0:51:28 
# Iter:   5200, Train Loss:  0.007, Train Acc: 100.00%, Val Loss:  0.081, Val Acc:  97.99%, Time: 0:52:29 *
# Iter:   5300, Train Loss:  0.085, Train Acc:  96.88%, Val Loss:  0.084, Val Acc:  97.86%, Time: 0:53:28 
# Iter:   5400, Train Loss: 0.0038, Train Acc: 100.00%, Val Loss:   0.13, Val Acc:  96.42%, Time: 0:54:29 
# Epoch:7
# Iter:   5500, Train Loss: 0.0031, Train Acc: 100.00%, Val Loss:  0.076, Val Acc:  98.15%, Time: 0:55:29 *
# Iter:   5600, Train Loss: 0.0054, Train Acc: 100.00%, Val Loss:  0.082, Val Acc:  97.99%, Time: 0:56:30 
# Iter:   5700, Train Loss: 0.0012, Train Acc: 100.00%, Val Loss:    0.1, Val Acc:  97.48%, Time: 0:57:30 
# Iter:   5800, Train Loss: 0.0011, Train Acc: 100.00%, Val Loss:  0.079, Val Acc:  97.92%, Time: 0:58:30 
# Iter:   5900, Train Loss: 0.00022, Train Acc: 100.00%, Val Loss:  0.085, Val Acc:  97.83%, Time: 0:59:31 
# Iter:   6000, Train Loss:  0.017, Train Acc:  98.44%, Val Loss:   0.13, Val Acc:  96.96%, Time: 1:00:31 
# Iter:   6100, Train Loss:  0.015, Train Acc:  98.44%, Val Loss:   0.11, Val Acc:  97.25%, Time: 1:01:31 
# Iter:   6200, Train Loss:  0.032, Train Acc:  98.44%, Val Loss:  0.092, Val Acc:  97.85%, Time: 1:02:32 
# Epoch:8
# Iter:   6300, Train Loss: 0.0048, Train Acc: 100.00%, Val Loss:  0.095, Val Acc:  97.90%, Time: 1:03:32 
# Iter:   6400, Train Loss: 0.0015, Train Acc: 100.00%, Val Loss:   0.11, Val Acc:  97.49%, Time: 1:04:33 
# Iter:   6500, Train Loss: 0.00079, Train Acc: 100.00%, Val Loss:   0.11, Val Acc:  97.64%, Time: 1:05:33 
# Iter:   6600, Train Loss: 0.00074, Train Acc: 100.00%, Val Loss:   0.11, Val Acc:  97.53%, Time: 1:06:34 
# Iter:   6700, Train Loss: 0.0013, Train Acc: 100.00%, Val Loss:   0.11, Val Acc:  97.66%, Time: 1:07:34 
# Iter:   6800, Train Loss: 0.00045, Train Acc: 100.00%, Val Loss:   0.11, Val Acc:  97.50%, Time: 1:08:35 
# Iter:   6900, Train Loss:   0.01, Train Acc: 100.00%, Val Loss:   0.11, Val Acc:  97.47%, Time: 1:09:35 
# Iter:   7000, Train Loss: 0.0035, Train Acc: 100.00%, Val Loss:   0.14, Val Acc:  96.99%, Time: 1:10:35 
# Epoch:9
# Iter:   7100, Train Loss: 0.00027, Train Acc: 100.00%, Val Loss:   0.12, Val Acc:  97.43%, Time: 1:11:35 
# Iter:   7200, Train Loss: 0.00061, Train Acc: 100.00%, Val Loss:  0.098, Val Acc:  97.87%, Time: 1:12:35 
# Iter:   7300, Train Loss: 0.00031, Train Acc: 100.00%, Val Loss:   0.11, Val Acc:  97.64%, Time: 1:13:36 
# Iter:   7400, Train Loss: 0.00077, Train Acc: 100.00%, Val Loss:   0.12, Val Acc:  97.43%, Time: 1:14:36 
# Iter:   7500, Train Loss: 0.00084, Train Acc: 100.00%, Val Loss:   0.14, Val Acc:  97.38%, Time: 1:15:37 
# 长时间未优化
```

测试结果

```

# Precision, Recall and F1-Score...
#               precision    recall  f1-score   support

#           体育       1.00      1.00      1.00      1000
#           娱乐       0.99      0.98      0.99      1000
#           家居       0.97      0.95      0.96      1000
#           房产       0.99      0.99      0.99      1000
#           教育       0.97      0.96      0.97      1000
#           时尚       0.99      0.98      0.98      1000
#           时政       0.96      0.98      0.97      1000
#           游戏       0.99      0.99      0.99      1000
#           科技       0.97      0.99      0.98      1000
#           财经       0.98      1.00      0.99      1000

#     accuracy                           0.98     10000
#    macro avg       0.98      0.98      0.98     10000
# weighted avg       0.98      0.98      0.98     10000

# Confusion Matrix...
# [[997   0   0   0   1   0   2   0   0   0]
#  [  0 983   3   0   6   0   1   4   1   2]
#  [  0   0 946   6   5   5  22   1   4  11]
#  [  0   0   5 990   0   1   1   0   0   3]
#  [  0   0   5   2 963   0  12   3  13   2]
#  [  0   4  15   0   5 976   0   0   0   0]
#  [  0   2   1   1   2   0 979   1   9   5]
#  [  0   0   0   0   3   1   0 994   2   0]
#  [  0   0   1   0   2   2   0   4 991   0]
#  [  0   0   0   0   1   0   2   0   1 996]]
```

实际应该用验证集测试的，不过这些都在colab里运行的，网速太慢，懒得上传数据了😂
