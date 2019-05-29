# TextRNN

## 数据
使用的是 THUCNews 的一个子集，下载地址：链接: https://pan.baidu.com/s/1hugrfRu 密码: qfud

(查看[TextCNN](/codes/textcnn))
(查看[HAN](/codes/han))

创建两个文件夹：data,output。将数据集中的 txt 文件放置在data目录下。output到时候用来保存训练好的模型

## 配置
`config.yaml`为配置文件，这只是简单的小例子，供学习 TextRNN 的原理使用的，完全没有调参。

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
6. 查看语料长度分布（可运行，也可不运行）

```python
plot_sentences_distribution("./data/cnews_final_train.txt")
```
7. 执行`train.py`中的 train 函数进行训练
8. 执行`train.py`中的 test 函数进行测试


## 结果
训练过程（实际运行打印不是如此，有些运行环境中会有大量tqdm进度信息，具体要显示什么输出自己在代码里改吧）
```
# Iter:    100, Train Loss:   0.54, Train Acc:  84.38%, Val Loss:   0.73, Val Acc:  79.43%, Time: 0:00:16 *
# Iter:    200, Train Loss:   0.29, Train Acc:  87.50%, Val Loss:   0.42, Val Acc:  88.33%, Time: 0:00:32 *
# Iter:    300, Train Loss:   0.41, Train Acc:  90.62%, Val Loss:   0.33, Val Acc:  91.21%, Time: 0:00:48 *
# Iter:    400, Train Loss:   0.26, Train Acc:  90.62%, Val Loss:   0.46, Val Acc:  88.02%, Time: 0:01:04 
# Iter:    500, Train Loss:    0.2, Train Acc:  95.31%, Val Loss:   0.32, Val Acc:  91.08%, Time: 0:01:21 
# Iter:    600, Train Loss:   0.24, Train Acc:  92.19%, Val Loss:   0.27, Val Acc:  92.41%, Time: 0:01:37 *
# Iter:    700, Train Loss:   0.75, Train Acc:  78.12%, Val Loss:   0.35, Val Acc:  90.13%, Time: 0:01:53 
# Iter:    800, Train Loss:    0.7, Train Acc:  89.06%, Val Loss:   0.83, Val Acc:  79.55%, Time: 0:02:09 
# Iter:    900, Train Loss:   0.26, Train Acc:  92.19%, Val Loss:   0.31, Val Acc:  91.41%, Time: 0:02:25 
# Iter:   1000, Train Loss:   0.21, Train Acc:  93.75%, Val Loss:   0.34, Val Acc:  91.20%, Time: 0:02:42 
# Iter:   1100, Train Loss:   0.32, Train Acc:  90.62%, Val Loss:   0.39, Val Acc:  89.13%, Time: 0:02:58 
# Iter:   1200, Train Loss:    0.2, Train Acc:  93.75%, Val Loss:   0.29, Val Acc:  91.74%, Time: 0:03:14 
# Iter:   1300, Train Loss:   0.34, Train Acc:  87.50%, Val Loss:   0.32, Val Acc:  90.91%, Time: 0:03:30 
# Iter:   1400, Train Loss:   0.31, Train Acc:  92.19%, Val Loss:   0.21, Val Acc:  93.72%, Time: 0:03:47 *
# Iter:   1500, Train Loss:   0.27, Train Acc:  90.62%, Val Loss:   0.24, Val Acc:  92.89%, Time: 0:04:04 
# Iter:   1600, Train Loss:  0.089, Train Acc:  96.88%, Val Loss:   0.21, Val Acc:  93.90%, Time: 0:04:20 *
# Iter:   1700, Train Loss:   0.31, Train Acc:  92.19%, Val Loss:    0.2, Val Acc:  94.59%, Time: 0:04:36 *
# Iter:   1800, Train Loss:   0.15, Train Acc:  98.44%, Val Loss:   0.18, Val Acc:  94.84%, Time: 0:04:52 *
# Iter:   1900, Train Loss:   0.26, Train Acc:  95.31%, Val Loss:   0.19, Val Acc:  94.60%, Time: 0:05:08 
# Iter:   2000, Train Loss:   0.13, Train Acc:  95.31%, Val Loss:   0.23, Val Acc:  92.93%, Time: 0:05:24 
# Iter:   2100, Train Loss:   0.17, Train Acc:  95.31%, Val Loss:    0.2, Val Acc:  94.22%, Time: 0:05:41 
# Iter:   2200, Train Loss:   0.15, Train Acc:  95.31%, Val Loss:   0.14, Val Acc:  95.96%, Time: 0:05:57 *
# Iter:   2300, Train Loss:   0.27, Train Acc:  90.62%, Val Loss:   0.17, Val Acc:  94.83%, Time: 0:06:12 
# Iter:   2400, Train Loss:   0.14, Train Acc:  92.19%, Val Loss:   0.18, Val Acc:  95.20%, Time: 0:06:28 
# Iter:   2500, Train Loss:  0.045, Train Acc:  98.44%, Val Loss:   0.17, Val Acc:  95.51%, Time: 0:06:45 
# Iter:   2600, Train Loss:   0.23, Train Acc:  90.62%, Val Loss:   0.23, Val Acc:  93.89%, Time: 0:07:01 
# Iter:   2700, Train Loss:  0.047, Train Acc:  98.44%, Val Loss:   0.16, Val Acc:  95.48%, Time: 0:07:17 
# Iter:   2800, Train Loss:   0.11, Train Acc:  95.31%, Val Loss:   0.13, Val Acc:  96.30%, Time: 0:07:33 *
# Iter:   2900, Train Loss:  0.083, Train Acc:  96.88%, Val Loss:   0.15, Val Acc:  95.69%, Time: 0:07:49 
# Iter:   3000, Train Loss:  0.031, Train Acc: 100.00%, Val Loss:   0.17, Val Acc:  94.92%, Time: 0:08:06 
# Iter:   3100, Train Loss:   0.14, Train Acc:  93.75%, Val Loss:   0.14, Val Acc:  96.12%, Time: 0:08:22 
# Iter:   3200, Train Loss:   0.21, Train Acc:  93.75%, Val Loss:   0.21, Val Acc:  93.41%, Time: 0:08:38 
# Iter:   3300, Train Loss:   0.21, Train Acc:  95.31%, Val Loss:   0.21, Val Acc:  94.38%, Time: 0:08:54 
# Iter:   3400, Train Loss:  0.083, Train Acc:  95.31%, Val Loss:   0.18, Val Acc:  94.67%, Time: 0:09:11 
# Iter:   3500, Train Loss:    0.1, Train Acc:  95.31%, Val Loss:   0.16, Val Acc:  95.57%, Time: 0:09:28 
# Iter:   3600, Train Loss:  0.049, Train Acc:  98.44%, Val Loss:   0.18, Val Acc:  94.88%, Time: 0:09:44 
# Iter:   3700, Train Loss:   0.16, Train Acc:  95.31%, Val Loss:   0.18, Val Acc:  94.87%, Time: 0:10:00 
# Iter:   3800, Train Loss:  0.049, Train Acc:  98.44%, Val Loss:   0.15, Val Acc:  95.70%, Time: 0:10:15 
# 长时间未优化
```

测试结果

```
# Precision, Recall and F1-Score...
#               precision    recall  f1-score   support

#           体育       1.00      1.00      1.00      1000
#           娱乐       0.97      0.99      0.98      1000
#           家居       0.95      0.88      0.91      1000
#           房产       0.94      0.98      0.96      1000
#           教育       0.98      0.92      0.95      1000
#           时尚       0.96      0.97      0.96      1000
#           时政       0.96      0.94      0.95      1000
#           游戏       0.97      0.98      0.97      1000
#           科技       0.96      0.97      0.97      1000
#           财经       0.96      1.00      0.98      1000

#     accuracy                           0.96     10000
#    macro avg       0.96      0.96      0.96     10000
# weighted avg       0.96      0.96      0.96     10000

# Confusion Matrix...
# [[998   0   0   0   0   0   0   2   0   0]
#  [  0 991   2   0   1   0   4   2   0   0]
#  [  1   9 877  45   6  24  18   4   5  11]
#  [  0   1   9 979   0   5   2   0   0   4]
#  [  4   6   8   2 922   4  11  14  16  13]
#  [  0  10  12   0   3 972   1   1   1   0]
#  [  0   5   2  10   9   0 945   1  13  15]
#  [  0   3   6   1   2   4   1 979   4   0]
#  [  0   1   7   2   0   8   1  10 971   0]
#  [  0   0   0   1   0   0   3   0   0 996]]
```

实际应该用验证集测试的，不过这些都在colab里运行的，网速太慢，懒得上传数据了😂
