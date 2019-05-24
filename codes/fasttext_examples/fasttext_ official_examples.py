#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-21 18:36:58
@author: wind
'''
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
print(*model.test(valid_data))
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