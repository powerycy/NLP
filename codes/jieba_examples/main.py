#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-14 16:23:33
@author: wind
'''

import jieba

# 分词
print("="*20,"分词","="*20)
seg_list = jieba.cut("我来到北京清华大学", cut_all=True)
print("全模式: " + "/ ".join(seg_list))  # 全模式

seg_list = jieba.cut("我来到北京清华大学", cut_all=False)
print("精确模式（默认）: " + "/ ".join(seg_list))  # 精确模式

seg_list = jieba.cut_for_search("小明硕士毕业于中国科学院计算所，后在日本京都大学深造")  # 搜索引擎模式
print("搜索引擎模式：" + "/ ".join(seg_list)) # 搜索引擎模式

# 加载词典
print("="*20,"加载词典","="*20)

test_sent = "李小福是创新办主任也是云计算方面的专家; 什么是八一双鹿"
words = jieba.cut(test_sent)
print("加载词典前：" + '/'.join(words))

jieba.load_userdict("userdict.txt")

words = jieba.cut(test_sent)
print("加载词典后：" + '/'.join(words))

# 动态调整词典
print("="*20,"动态调整词典","="*20)

print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))
jieba.suggest_freq(('中', '将'), True)
print('/'.join(jieba.cut('如果放到post中将出错。', HMM=False)))

print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))
jieba.suggest_freq('台中', True)
print('/'.join(jieba.cut('「台中」正确应该不会被切开', HMM=False)))


# 全模式: 我/ 来到/ 北京/ 清华/ 清华大学/ 华大/ 大学
# 精确模式（默认）: 我/ 来到/ 北京/ 清华大学
# 搜索引擎模式：小明/ 硕士/ 毕业/ 于/ 中国/ 科学/ 学院/ 科学院/ 中国科学院/ 计算/ 计算所/ ，/ 后/ 在/ 日本/ 京都/ 大学/ 日本京都大学/ 深造



# 词性标注
print("="*20,"词性标注","="*20)
import jieba.posseg as pseg
words = pseg.cut("我爱北京天安门")
for word, flag in words:
    print('%s %s' % (word, flag))

# tokenize
print("="*20,"tokenize","="*20)

print("默认模式:")
result = jieba.tokenize('永和服装饰品有限公司')
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))

print("搜索模式:")
result = jieba.tokenize(u'永和服装饰品有限公司', mode='search')
for tk in result:
    print("word %s\t\t start: %d \t\t end:%d" % (tk[0],tk[1],tk[2]))
