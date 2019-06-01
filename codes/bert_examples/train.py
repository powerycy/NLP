#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-24 17:29:27
@author: wind
'''
import os
import time
from datetime import timedelta
from sklearn import metrics
from tqdm import tqdm,trange
import numpy as np
import torch
from datahelper import NewsDataset,get_labels
from config import Config
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam,BertConfig


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def get_model_labels(model,word_ids,segment_ids,word_mask):
    logits = model.forward(word_ids,segment_ids,word_mask)
    labels = torch.argmax(logits,1)
    labels = labels.to("cpu").numpy()
    return labels

def get_model_loss_acc(model,word_ids,segment_ids,word_mask,labels):
    logits = model.forward(word_ids,segment_ids,word_mask)

    loss_fct = CrossEntropyLoss()
    loss = loss_fct(logits, labels)

    eq = torch.argmax(logits,1) == labels
    eq = eq.to("cpu").numpy()
    acc = eq.sum()/len(eq)

    return loss,acc


def evaluate(model, test_dataloader,device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    data_len = 0
    # for batch in tqdm(test_dataloader,"评估",total=len(test_dataloader)):
    for batch in test_dataloader:
        
        label_id = batch['label_id'].squeeze(1).to(device) 
        word_ids = batch['word_ids'].to(device) 
        segment_ids = batch['segment_ids'].to(device) 
        word_mask = batch['word_mask'].to(device) 

        with torch.no_grad():
            loss,acc = get_model_loss_acc(model,word_ids,segment_ids,word_mask,label_id)
        
        batch_len = label_id.size(0)
        data_len += batch_len
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    
    return total_loss / data_len, total_acc / data_len
                
def train_fixed():
    # 配置文件
    cf = Config('./config.yaml')
    # 有GPU用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练数据
    train_data = NewsDataset("./data/cnews_final_train.txt",cf.max_seq_len)
    train_dataloader = DataLoader(train_data,batch_size=cf.batch_size,shuffle=True)
    # 测试数据
    test_data = NewsDataset("./data/cnews_final_test.txt",cf.max_seq_len)
    test_dataloader = DataLoader(test_data,batch_size=cf.batch_size,shuffle=True)
    # 模型
    if os.path.isfile("./bert/bert-base-chinese.tar.gz"):
        model = BertForSequenceClassification.from_pretrained('./bert/bert-base-chinese.tar.gz',num_labels=cf.num_labels)
    else:
        model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=cf.num_labels)
    
    # 优化器用adam
    for param in model.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True
    # param_optimizer = list(model.named_parameters())
    # no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    # optimizer_grouped_parameters = [
    #     {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    #     {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        
    # num_train_optimization_steps = int(len(train_data) / cf.batch_size)*cf.epoch
    # optimizer = BertAdam(optimizer_grouped_parameters,lr=cf.lr,
    #                     t_total=num_train_optimization_steps)

    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()))

    # 把模型放到指定设备
    model.to(device)

    # 让模型并行化运算
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
    
    # 训练
    start_time = time.time()

    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1500  # 如果超过1500轮未提升，提前结束训练

    flag = False
    model.train()
    for epoch_id in range(cf.epoch):
        print("Epoch %d"%epoch_id)
        for step,batch in enumerate(tqdm(train_dataloader,desc="batch",total=len(train_dataloader))):
        # for step,batch in enumerate(train_dataloader):
            
            label_id = batch['label_id'].squeeze(1).to(device) 
            word_ids = batch['word_ids'].to(device) 
            segment_ids = batch['segment_ids'].to(device) 
            word_mask = batch['word_mask'].to(device) 

            loss = model(word_ids,segment_ids,word_mask,label_id)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_batch += 1 

            if total_batch % cf.print_per_batch == 0:
                model.eval()
                with torch.no_grad():
                    loss_train,acc_train = get_model_loss_acc(model,word_ids,segment_ids,word_mask,label_id)
                loss_val,acc_val = evaluate(model,test_dataloader,device)
                
                if acc_val  > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch

                    torch.save(model.state_dict(),"./output/pytorch_model.bin")
                    with open("./output/pytorch_bert_config.json",'w') as f:
                        f.write(model.config.to_json_string())

                    improved_str = "*"
                else:
                    improved_str = ""
                
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
                
                model.train()

            if total_batch - last_improved > require_improvement:
                print("长时间未优化")
                flag = True
                break
        if flag:
            break


def train_unfixed():
    # 配置文件
    cf = Config('./config.yaml')
    # 有GPU用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 训练数据
    train_data = NewsDataset("./data/cnews_final_train.txt",cf.max_seq_len)
    train_dataloader = DataLoader(train_data,batch_size=cf.batch_size,shuffle=True)
    # 测试数据
    test_data = NewsDataset("./data/cnews_final_test.txt",cf.max_seq_len)
    test_dataloader = DataLoader(test_data,batch_size=cf.batch_size,shuffle=True)
    
    # 模型
    config = BertConfig("./output/pytorch_bert_config.json")
    model = BertForSequenceClassification(config,num_labels=cf.num_labels)
    model.load_state_dict(torch.load("./output/pytorch_model.bin"))

    # 优化器用adam
    for param in model.parameters():
        param.requires_grad = True
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
        
    num_train_optimization_steps = int(len(train_data) / cf.batch_size)*cf.epoch
    optimizer = BertAdam(optimizer_grouped_parameters,lr=cf.lr,
                        t_total=num_train_optimization_steps)

    # 把模型放到指定设备
    model.to(device)

    # 让模型并行化运算
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
    
    # 训练
    start_time = time.time()

    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    last_improved = 0  # 记录上一次提升批次
    require_improvement = 1500  # 如果超过1500轮未提升，提前结束训练

    # 获取当前验证集acc
    model.eval()
    _,best_acc_val = evaluate(model,test_dataloader,device)

    flag = False
    model.train()
    for epoch_id in range(cf.epoch):
        print("Epoch %d"%epoch_id)
        for step,batch in enumerate(tqdm(train_dataloader,desc="batch",total=len(train_dataloader))):
        # for step,batch in enumerate(train_dataloader):
            
            label_id = batch['label_id'].squeeze(1).to(device) 
            word_ids = batch['word_ids'].to(device) 
            segment_ids = batch['segment_ids'].to(device) 
            word_mask = batch['word_mask'].to(device) 

            loss = model(word_ids,segment_ids,word_mask,label_id)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_batch += 1 

            if total_batch % cf.print_per_batch == 0:
                model.eval()
                with torch.no_grad():
                    loss_train,acc_train = get_model_loss_acc(model,word_ids,segment_ids,word_mask,label_id)
                loss_val,acc_val = evaluate(model,test_dataloader,device)
                
                if acc_val  > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch

                    torch.save(model.state_dict(),"./output/pytorch_model.bin")
                    with open("./output/pytorch_bert_config.json",'w') as f:
                        f.write(model.config.to_json_string())

                    improved_str = "*"
                else:
                    improved_str = ""
                
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))
                
                model.train()

            if total_batch - last_improved > require_improvement:
                print("长时间未优化")
                flag = True
                break
        if flag:
            break

def test():
    # 配置文件
    cf = Config('./config.yaml')
    # 有GPU用GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 测试数据
    test_data = NewsDataset("./data/cnews_final_test.txt",cf.max_seq_len)
    test_dataloader = DataLoader(test_data,batch_size=cf.batch_size,shuffle=True)

    # 模型
    config = BertConfig("./output/pytorch_bert_config.json")
    model = BertForSequenceClassification(config,num_labels=cf.num_labels)
    model.load_state_dict(torch.load("./output/pytorch_model.bin"))

    # 把模型放到指定设备
    model.to(device)

    # 让模型并行化运算
    if torch.cuda.device_count()>1:
        model = torch.nn.DataParallel(model)
    
    # 训练
    start_time = time.time()

    data_len = len(test_dataloader)

    model.eval()
    y_pred = np.array([])
    y_test = np.array([])
    # for step,batch in enumerate(tqdm(test_dataloader,"batch",total=len(test_dataloader))):
    for step,batch in enumerate(test_dataloader):
        label_id = batch['label_id'].squeeze(1).to(device) 
        word_ids = batch['word_ids'].to(device) 
        segment_ids = batch['segment_ids'].to(device) 
        word_mask = batch['word_mask'].to(device) 

        loss = model(word_ids,segment_ids,word_mask,label_id)
        
        with torch.no_grad():
            pred = get_model_labels(model,word_ids,segment_ids,word_mask)
        y_pred = np.hstack((y_pred,pred))
        y_test = np.hstack((y_test,label_id.to("cpu").numpy()))

    # 评估
    print("Precision, Recall and F1-Score...")
    print(metrics.classification_report(y_test, y_pred, target_names=get_labels('./data/label')))

    # 混淆矩阵
    print("Confusion Matrix...")
    cm = metrics.confusion_matrix(y_test, y_pred)
    print(cm)


if __name__ == "__main__":
    # train_fixed()
    train_unfixed()
    # test()

# train_fixed()输出
# Iter:    300, Train Loss:   0.94, Train Acc:  80.00%, Val Loss:   0.87, Val Acc:  74.62%, Time: 0:01:43 *
# Iter:    600, Train Loss:   0.28, Train Acc:  90.00%, Val Loss:    0.7, Val Acc:  81.24%, Time: 0:03:26 *
# Iter:    900, Train Loss:   0.15, Train Acc:  90.00%, Val Loss:   0.54, Val Acc:  83.54%, Time: 0:05:09 *
# Iter:   1200, Train Loss:   0.63, Train Acc:  90.00%, Val Loss:   0.53, Val Acc:  84.10%, Time: 0:06:53 *
# Iter:   1500, Train Loss:   0.15, Train Acc: 100.00%, Val Loss:   0.46, Val Acc:  85.36%, Time: 0:08:35 *
# Iter:   1800, Train Loss:   0.29, Train Acc:  90.00%, Val Loss:   0.46, Val Acc:  86.13%, Time: 0:10:19 *
# Iter:   2100, Train Loss:   0.27, Train Acc:  90.00%, Val Loss:    0.4, Val Acc:  88.03%, Time: 0:12:02 *
# Iter:   2400, Train Loss:  0.071, Train Acc: 100.00%, Val Loss:   0.41, Val Acc:  87.17%, Time: 0:13:43 
# Iter:   2700, Train Loss:  0.005, Train Acc: 100.00%, Val Loss:   0.47, Val Acc:  85.77%, Time: 0:15:24 
# Iter:   3000, Train Loss:  0.028, Train Acc: 100.00%, Val Loss:    0.4, Val Acc:  87.74%, Time: 0:17:05 
# Iter:   3300, Train Loss:   0.37, Train Acc:  90.00%, Val Loss:   0.48, Val Acc:  85.58%, Time: 0:18:46 
# Iter:   3600, Train Loss:   0.21, Train Acc: 100.00%, Val Loss:   0.37, Val Acc:  88.25%, Time: 0:20:28 *
# Iter:   3900, Train Loss:  0.086, Train Acc: 100.00%, Val Loss:   0.44, Val Acc:  86.72%, Time: 0:22:09 
# Iter:   4200, Train Loss:   0.13, Train Acc:  90.00%, Val Loss:   0.45, Val Acc:  86.13%, Time: 0:23:50 
# Iter:   4500, Train Loss:   0.21, Train Acc:  90.00%, Val Loss:   0.34, Val Acc:  88.94%, Time: 0:25:32 *
# Iter:   4800, Train Loss:  0.041, Train Acc: 100.00%, Val Loss:    0.3, Val Acc:  90.53%, Time: 0:27:15 *
# Iter:   5100, Train Loss:   0.32, Train Acc:  90.00%, Val Loss:   0.32, Val Acc:  89.45%, Time: 0:28:56 
# Iter:   5400, Train Loss:  0.064, Train Acc: 100.00%, Val Loss:   0.33, Val Acc:  89.38%, Time: 0:30:37 
# Iter:   5700, Train Loss:   0.31, Train Acc:  80.00%, Val Loss:   0.34, Val Acc:  89.13%, Time: 0:32:17 
# Iter:   6000, Train Loss:  0.021, Train Acc: 100.00%, Val Loss:   0.38, Val Acc:  88.00%, Time: 0:33:58 
# Iter:   6300, Train Loss:   0.24, Train Acc:  90.00%, Val Loss:    0.3, Val Acc:  90.30%, Time: 0:35:39 
# 长时间未优化


# train_unfixed()输出
# Iter:    300, Train Loss:  0.048, Train Acc: 100.00%, Val Loss:   0.65, Val Acc:  86.70%, Time: 0:04:02 
# Iter:    600, Train Loss:  0.041, Train Acc: 100.00%, Val Loss:   0.43, Val Acc:  89.79%, Time: 0:06:44 
# Iter:    900, Train Loss: 0.0073, Train Acc: 100.00%, Val Loss:   0.37, Val Acc:  91.26%, Time: 0:09:28 *
# Iter:   1200, Train Loss:  0.036, Train Acc: 100.00%, Val Loss:    0.3, Val Acc:  92.78%, Time: 0:12:12 *
# Iter:   1500, Train Loss:   0.45, Train Acc:  90.00%, Val Loss:   0.29, Val Acc:  92.87%, Time: 0:14:56 *
# Iter:   1800, Train Loss:  0.091, Train Acc:  90.00%, Val Loss:   0.27, Val Acc:  94.13%, Time: 0:17:40 *
# Iter:   2100, Train Loss:   0.75, Train Acc:  90.00%, Val Loss:   0.35, Val Acc:  93.06%, Time: 0:20:23 
# Iter:   2400, Train Loss: 0.0042, Train Acc: 100.00%, Val Loss:    0.3, Val Acc:  93.39%, Time: 0:23:05 
# Iter:   2700, Train Loss:  0.031, Train Acc: 100.00%, Val Loss:   0.28, Val Acc:  93.03%, Time: 0:25:47 
# Iter:   3000, Train Loss:  0.015, Train Acc: 100.00%, Val Loss:   0.26, Val Acc:  93.75%, Time: 0:28:29 
# Iter:   3300, Train Loss:    1.2, Train Acc:  80.00%, Val Loss:   0.25, Val Acc:  94.40%, Time: 0:31:13 *
# Iter:   3600, Train Loss: 0.0017, Train Acc: 100.00%, Val Loss:   0.23, Val Acc:  94.93%, Time: 0:33:57 *
# Iter:   3900, Train Loss: 0.0067, Train Acc: 100.00%, Val Loss:   0.24, Val Acc:  94.88%, Time: 0:36:40 
# Iter:   4200, Train Loss:  0.005, Train Acc: 100.00%, Val Loss:   0.23, Val Acc:  93.90%, Time: 0:39:22 
# Iter:   4500, Train Loss:  0.014, Train Acc: 100.00%, Val Loss:   0.24, Val Acc:  95.38%, Time: 0:42:07 *
# Iter:   4800, Train Loss:   0.61, Train Acc:  90.00%, Val Loss:   0.25, Val Acc:  94.29%, Time: 0:44:49 
# Iter:   5100, Train Loss:   0.38, Train Acc:  90.00%, Val Loss:    0.3, Val Acc:  93.61%, Time: 0:47:32 
# Iter:   5400, Train Loss: 0.0019, Train Acc: 100.00%, Val Loss:   0.32, Val Acc:  94.26%, Time: 0:50:14 
# Iter:   5700, Train Loss:  0.045, Train Acc: 100.00%, Val Loss:   0.26, Val Acc:  95.08%, Time: 0:52:56 
# Iter:   6000, Train Loss: 0.0036, Train Acc: 100.00%, Val Loss:   0.24, Val Acc:  95.00%, Time: 0:55:38 
# 长时间未优化

# test()输出
# Precision, Recall and F1-Score...
#               precision    recall  f1-score   support

#           体育       1.00      0.99      0.99      1000
#           娱乐       0.98      0.97      0.98      1000
#           家居       0.97      0.84      0.90      1000
#           房产       0.90      0.86      0.88      1000
#           教育       0.94      0.95      0.95      1000
#           时尚       0.95      0.99      0.97      1000
#           时政       0.92      0.98      0.95      1000
#           游戏       0.96      0.98      0.97      1000
#           科技       0.97      0.98      0.98      1000
#           财经       0.95      0.98      0.97      1000

#     accuracy                           0.95     10000
#    macro avg       0.95      0.95      0.95     10000
# weighted avg       0.95      0.95      0.95     10000

# Confusion Matrix...
# [[989   0   0   0   0   0   2   8   0   1]
#  [  0 974   0   0   5   7   2  10   2   0]
#  [  2   9 845  68  12  30  16   3   4  11]
#  [  0   4  21 857  33   8  51   1   3  22]
#  [  0   0   0   4 951   4   8  11  15   7]
#  [  0   1   1   0   4 993   0   0   1   0]
#  [  0   1   0   9   1   1 977   2   5   4]
#  [  0   5   0   0   4   4   0 982   4   1]
#  [  0   0   2   1   1   2   1   6 985   2]
#  [  0   0   0  12   1   0   2   0   0 985]]