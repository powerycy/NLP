# TextCNN
 
## 数据
使用的是 THUCNews 的一个子集，下载地址：链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud

(查看[TextRNN](/codes/textrnn))

创建两个文件夹：data,output。将数据集中的 txt 文件放置在data目录下。output到时候用来保存训练好的模型

## 配置

`config.yaml`为配置文件，这只是简单的小例子，供学习 TextCNN 的原理使用的，完全没有调参，而且里面设置了个很大的`max_seq_len`，因为大部分文章长度在2000-4000之间，还有极个别的文章长度上万。

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
3. 保留常用的5000个词的词向量作为词典
```python
 top_k_vec("./data/vectors","./data/stop_words.txt","./data/final_vectors",5000)
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
6. 执行`train.py`中的 train 函数进行训练
7. 执行`train.py`中的 test 函数进行测试


在 [TextRNN](/codes/textrnn) 中去掉了不常用的词汇，在`predata.py`中加了
```python
plot_sentences_distribution("./data/cnews_final_train.txt")
```
来查看文章长度的分布

## 结果
训练过程（实际运行打印不是如此，有些运行环境中会有大量tqdm进度信息，具体要显示什么输出自己在代码里改吧）
```
# Iter:    100, Train Loss:    0.2, Train Acc:  90.62%, Val Loss:   0.23, Val Acc:  93.27%, Time: 0:01:11 *
# Iter:    200, Train Loss:   0.18, Train Acc:  93.75%, Val Loss:   0.22, Val Acc:  93.22%, Time: 0:02:22 
# Iter:    300, Train Loss:  0.094, Train Acc:  96.88%, Val Loss:   0.16, Val Acc:  95.03%, Time: 0:03:33 *
# Iter:    400, Train Loss:   0.12, Train Acc:  96.88%, Val Loss:   0.15, Val Acc:  95.43%, Time: 0:04:44 *
# Iter:    500, Train Loss:  0.079, Train Acc:  96.88%, Val Loss:   0.15, Val Acc:  95.53%, Time: 0:05:55 *
# Iter:    600, Train Loss:  0.026, Train Acc: 100.00%, Val Loss:   0.14, Val Acc:  95.61%, Time: 0:07:06 *
# Iter:    700, Train Loss:  0.046, Train Acc:  98.44%, Val Loss:   0.16, Val Acc:  95.33%, Time: 0:08:17 
# Iter:    800, Train Loss:    0.1, Train Acc:  98.44%, Val Loss:   0.17, Val Acc:  95.38%, Time: 0:09:27 
# Iter:    900, Train Loss:  0.026, Train Acc: 100.00%, Val Loss:   0.18, Val Acc:  94.87%, Time: 0:10:38 
# Iter:   1000, Train Loss:   0.13, Train Acc:  96.88%, Val Loss:   0.17, Val Acc:  94.72%, Time: 0:11:49 
# Iter:   1100, Train Loss:   0.16, Train Acc:  95.31%, Val Loss:   0.17, Val Acc:  95.56%, Time: 0:13:00 
# Iter:   1200, Train Loss:  0.025, Train Acc:  98.44%, Val Loss:   0.12, Val Acc:  96.80%, Time: 0:14:11 *
# Iter:   1300, Train Loss:  0.054, Train Acc:  98.44%, Val Loss:   0.16, Val Acc:  95.57%, Time: 0:15:22 
# Iter:   1400, Train Loss:    0.2, Train Acc:  95.31%, Val Loss:   0.15, Val Acc:  95.89%, Time: 0:16:33 
# Iter:   1500, Train Loss:   0.27, Train Acc:  93.75%, Val Loss:   0.14, Val Acc:  96.24%, Time: 0:17:44 
# Iter:   1600, Train Loss:  0.013, Train Acc: 100.00%, Val Loss:   0.13, Val Acc:  96.09%, Time: 0:18:54 
# Iter:   1700, Train Loss:  0.036, Train Acc:  98.44%, Val Loss:   0.12, Val Acc:  96.56%, Time: 0:20:05 
# Iter:   1800, Train Loss:   0.18, Train Acc:  92.19%, Val Loss:   0.13, Val Acc:  96.24%, Time: 0:21:16 
# Iter:   1900, Train Loss:  0.037, Train Acc: 100.00%, Val Loss:   0.14, Val Acc:  96.17%, Time: 0:22:27 
# Iter:   2000, Train Loss:  0.035, Train Acc:  98.44%, Val Loss:   0.13, Val Acc:  96.59%, Time: 0:23:38 
# Iter:   2100, Train Loss:  0.036, Train Acc:  98.44%, Val Loss:   0.11, Val Acc:  97.07%, Time: 0:24:49 *
# Iter:   2200, Train Loss:  0.007, Train Acc: 100.00%, Val Loss:   0.13, Val Acc:  96.39%, Time: 0:26:00 
# Iter:   2300, Train Loss:   0.11, Train Acc:  95.31%, Val Loss:   0.12, Val Acc:  96.55%, Time: 0:27:11 
# Iter:   2400, Train Loss:  0.045, Train Acc:  98.44%, Val Loss:   0.12, Val Acc:  96.83%, Time: 0:28:21 
# Iter:   2500, Train Loss:  0.029, Train Acc: 100.00%, Val Loss:   0.11, Val Acc:  97.07%, Time: 0:29:32 
# Iter:   2600, Train Loss:  0.082, Train Acc:  95.31%, Val Loss:   0.13, Val Acc:  96.46%, Time: 0:30:43 
# Iter:   2700, Train Loss:  0.085, Train Acc:  95.31%, Val Loss:   0.11, Val Acc:  96.91%, Time: 0:31:54 
# Iter:   2800, Train Loss:  0.085, Train Acc:  96.88%, Val Loss:  0.099, Val Acc:  97.24%, Time: 0:33:05 *
# Iter:   2900, Train Loss:  0.059, Train Acc:  98.44%, Val Loss:   0.15, Val Acc:  95.91%, Time: 0:34:16 
# Iter:   3000, Train Loss:  0.064, Train Acc:  96.88%, Val Loss:   0.12, Val Acc:  96.73%, Time: 0:35:27 
# Iter:   3100, Train Loss:  0.016, Train Acc: 100.00%, Val Loss:   0.13, Val Acc:  96.20%, Time: 0:36:38 
# Iter:   3200, Train Loss:  0.067, Train Acc:  96.88%, Val Loss:   0.11, Val Acc:  97.00%, Time: 0:37:48 
# Iter:   3300, Train Loss:    0.1, Train Acc:  95.31%, Val Loss:   0.15, Val Acc:  96.08%, Time: 0:38:59 
# Iter:   3400, Train Loss:   0.02, Train Acc: 100.00%, Val Loss:   0.16, Val Acc:  96.02%, Time: 0:40:10 
# Iter:   3500, Train Loss: 0.0023, Train Acc: 100.00%, Val Loss:   0.12, Val Acc:  96.82%, Time: 0:41:21 
# Iter:   3600, Train Loss:   0.03, Train Acc:  98.44%, Val Loss:   0.12, Val Acc:  96.93%, Time: 0:42:32 
# Iter:   3700, Train Loss:  0.068, Train Acc:  96.88%, Val Loss:   0.12, Val Acc:  96.98%, Time: 0:43:43 
# Iter:   3800, Train Loss: 0.0077, Train Acc: 100.00%, Val Loss:   0.13, Val Acc:  96.58%, Time: 0:44:54 
# 长时间未优化
```

测试结果

```
# Precision, Recall and F1-Score...
#               precision    recall  f1-score   support

#           体育       1.00      1.00      1.00      1000
#           娱乐       0.99      0.98      0.99      1000
#           家居       0.97      0.92      0.95      1000
#           房产       1.00      0.99      1.00      1000
#           教育       0.95      0.96      0.96      1000
#           时尚       0.95      0.99      0.97      1000
#           时政       0.97      0.96      0.96      1000
#           游戏       0.99      0.95      0.97      1000
#           科技       0.95      0.97      0.96      1000
#           财经       0.95      0.99      0.97      1000

#     accuracy                           0.97     10000
#    macro avg       0.97      0.97      0.97     10000
# weighted avg       0.97      0.97      0.97     10000

# Confusion Matrix...
# [[997   0   0   0   0   0   1   0   1   1]
#  [  0 977   4   0   9   3   0   0   6   1]
#  [  1   2 919   2  13  16  11   2  11  23]
#  [  0   1   2 995   0   0   0   0   0   2]
#  [  0   1   5   0 964   2  10   0  10   8]
#  [  0   1   4   0   2 991   0   1   1   0]
#  [  1   0   2   1  12   1 961   1  10  11]
#  [  0   0   2   0   9  19   2 952  15   1]
#  [  0   0   5   0   2   6   3   8 973   3]
#  [  0   0   0   0   0   0   5   0   0 995]]
```

实际应该用验证集测试的，不过这些都在colab里运行的，网速太慢，懒得上传数据了😂
