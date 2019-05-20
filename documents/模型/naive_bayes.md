# 朴素贝叶斯

## 基本概念
先明确几个概念：

* 先验概率$P(A)$：在不考虑任何情况下，A时间发生的概率
* 条件概率$P(B|A)$：A事件发生的情况下，B事件发生的概率
* 后验概率$P(A|B)$：在B事件发生之后，对A事件发生的概率的重新评估
* 全概率：如果$A_1,A_2,\cdots,A_n$构成样本空间的一个划分，那么事件B的概率为：

$$
P(B) = \sum_{i=1}^n P(A_i) \cdot P(B|A_i)
$$


条件概率公式：

$$P(B|A) = \frac{P(AB)}{P(B)}$$

概率的乘法定理或乘法规则：

$$P(AB) = P(B)\cdot P(A|B) = P(A) \cdot P(B|A)$$

贝叶斯公式：

$$
P(A_i|B) = \frac{P(A_i)\cdot P(B|A_i)}{P(B)} = \frac{P(A_i)\cdot P(B|A_i)}{\sum_{i=1}^n P(A_i) \cdot P(B|A_i)}
$$

## 朴素贝叶斯原理

朴素贝叶斯是基于“特征之间是独立的”这一朴素假设，应用贝叶斯定理的监督学习算法

对应给定的样本$x$的特征向量$x_1,x_2,\cdots,x_m$,该样本$x$的类别$y$的概率可以由贝叶斯公式得到：

$$
P(y|x_1,x_2,\cdots,x_m) = \frac{P(y)P(x_1,x_2,\cdots,x_m|y)}{P(x_1,x_2,\cdots,x_m)}
$$

由于特征独立的假设，所以有：

$$
P(x_i|y,x_1,x_2,\cdots,x_{i-1},x_{i+1}\cdots,x_m) = P(x_i|y)
$$

所以：

$$
P(y|x_1,x_2,\cdots,x_m) = \frac{P(y)P(x_1,x_2,\cdots,x_m|y)}{P(x_1,x_2,\cdots,x_m)} = \frac{P(y)\prod_{i=1}^mP(x_i|y)}{P(x_1,x_2,\cdots,x_m)} 
$$

在给定样本下，$P(x_1,x_2,\cdots,x_m)$是常数，所以：

$$
P(y|x_1,x_2,\cdots,x_m) \propto P(y)\prod_{i=1}^mP(x_i|y)
$$

从而：

$$
\hat{y} = \mathop{\arg\min}_{a} P(y)\prod_{i=1}^mP(x_i|y)
$$

## 朴素贝叶斯算法流程

设$x=\{a_1,a_2,\cdots,a_m \}$ 为待分类项，其中$a_i$为$x$的一个特征属性。类别集合为$C=\{y_1,y_2,\cdots,y_n\}$

1. 计算每个类别概率：$P(y_1),P(y_2),\cdots,P(y_n)$
2. 对每个特征属性计算所有划分的条件概率：$P(x_{ij}|y_k)$
3. 对每个类别计算$P(y_i)\cdot P(x|y_i)$
4. 让$P(y_i)\cdot P(x|y_i)$最大的$y_i$即为$x$所属类别


如何求某个类别下，某个特征取某个特定值时的概率呢？

### 高斯朴素贝叶斯
### 伯努利朴素贝叶斯
### 多项式朴素贝叶斯

## 代码实现

参考北风ppt和机器学习实战以完善内容



---
**参考**：
1. [美] Peter Harrington 著，李锐 等 译 《机器学习实战》 人民邮电出版社
2. 刘建平博客：[朴素贝叶斯算法原理小结](https://www.cnblogs.com/pinard/p/6069267.html)