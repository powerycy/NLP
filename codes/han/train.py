#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-24 17:29:27
@author: wind
'''

import time
from datetime import timedelta
from sklearn import metrics
import numpy as np
from tqdm import tqdm,trange
import torch
from model import Config,TextRNN
from datahelper import NewsDataset,get_pre_embedding_matrix,get_labels
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam



def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


def evaluate(model, test_dataloader,device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    data_len = 0
    # for batch in tqdm(test_dataloader,"评估",total=len(test_dataloader)):
    for batch in test_dataloader:
        
        label_id = batch['label_id'].squeeze(1).to(device) 
        seq_len = batch["seq_len"].to(device) 
        segment_ids = batch['segment_ids'].to(device) 

        # 将序列按长度降序排列
        seq_len,perm_idx = seq_len.sort(0,descending=True)
        label_id = label_id[perm_idx]
        segment_ids = segment_ids[perm_idx].transpose(0,1)

        with torch.no_grad():
            loss,acc = model.get_loss_acc(segment_ids,seq_len,label_id)
        
        batch_len = label_id.size(0)
        data_len += batch_len
        total_loss += loss * batch_len
        total_acc += acc * batch_len
    
    return total_loss / data_len, total_acc / data_len
                

def train():
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

    # 预训练词向量矩阵
    embedding_matrix = get_pre_embedding_matrix("./data/final_vectors")
    # 模型
    model = TextRNN(cf,torch.tensor(embedding_matrix))
    # 优化器用adam
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
    require_improvement = 1000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    model.train()
    for epoch_id in trange(cf.epoch,desc="Epoch"):
        # for step,batch in enumerate(tqdm(train_dataloader,"batch",total=len(train_dataloader))):
        for step,batch in enumerate(train_dataloader):
            
            label_id = batch['label_id'].squeeze(1).to(device) 
            seq_len = batch["seq_len"].to(device) 
            segment_ids = batch['segment_ids'].to(device) 

            # 将序列按长度降序排列
            seq_len,perm_idx = seq_len.sort(0,descending=True)
            label_id = label_id[perm_idx]
            segment_ids = segment_ids[perm_idx].transpose(0,1)

            loss = model(segment_ids,seq_len,label_id)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_batch += 1 

            if total_batch % cf.print_per_batch == 0:
                model.eval()
                with torch.no_grad():
                    loss_train,acc_train = model.get_loss_acc(segment_ids,seq_len,label_id)
                loss_val,acc_val = evaluate(model,test_dataloader,device)
                
                if acc_val  > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    last_improved = total_batch
                    torch.save(model.state_dict(),"./output/model.bin")
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

    # 预训练词向量矩阵
    embedding_matrix = get_pre_embedding_matrix("./data/final_vectors")
    # 模型
    model = TextRNN(cf,torch.tensor(embedding_matrix))

    # model.load_state_dict(torch.load("./output/model.bin",map_location='cpu'))
    model.load_state_dict(torch.load("./output/model.bin"))
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
        seq_len = batch["seq_len"].to(device) 
        segment_ids = batch['segment_ids'].to(device) 

        # 将序列按长度降序排列
        seq_len,perm_idx = seq_len.sort(0,descending=True)
        label_id = label_id[perm_idx]
        segment_ids = segment_ids[perm_idx].transpose(0,1)

        with torch.no_grad():
            pred = model.get_labels(segment_ids,seq_len)
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
    # train()
    test()

# train()输出
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
            
# test()输出
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