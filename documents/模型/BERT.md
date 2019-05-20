# BERT

BERT的全称是Bidirectional Encoder Representation from Transformers，即双向Transformer的Encoder，因为decoder是不能获要预测的信息的。模型的主要创新点都在pre-train方法上，即用了Masked LM和Next Sentence Prediction两种方法分别捕捉词语和句子级别的representation。




---
**参考**：
1. 论文：[BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805)
2. 谷歌官方代码：[bert](https://github.com/google-research/bert)
3. pytorch版本代码：[pytorch-pretrained-BERT](https://github.com/huggingface/pytorch-pretrained-BERT)
4. [【NLP】Google BERT详解](https://zhuanlan.zhihu.com/p/46652512)
