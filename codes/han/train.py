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
# from tqdm import tqdm,trange
import torch
from model import Config,TextHAN
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
        lengths = batch["lengths"].to(device) 
        x = batch['x'].to(device) 

        with torch.no_grad():
            loss,acc = model.get_loss_acc(x,lengths,label_id)
        
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
    train_data = NewsDataset("./data/cnews_final_train.txt",cf.max_word_len,cf.max_sen_len)
    train_dataloader = DataLoader(train_data,batch_size=cf.batch_size,shuffle=True)
    # 测试数据
    test_data = NewsDataset("./data/cnews_final_test.txt",cf.max_word_len,cf.max_sen_len)
    test_dataloader = DataLoader(test_data,batch_size=cf.batch_size,shuffle=True)

    # 预训练词向量矩阵
    embedding_matrix = get_pre_embedding_matrix("./data/final_vectors")
    # 模型
    model = TextHAN(cf,torch.tensor(embedding_matrix))
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
    require_improvement = 2000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    model.train()
    # for epoch_id in trange(cf.epoch,desc="Epoch"):
    for epoch_id in range(cf.epoch):
        print("Epoch:%d"%epoch_id)
        # for step,batch in enumerate(tqdm(train_dataloader,"batch",total=len(train_dataloader))):
        for step,batch in enumerate(train_dataloader):
            
            label_id = batch['label_id'].squeeze(1).to(device) 
            lengths = batch["lengths"].to(device) 
            x = batch['x'].to(device) 

            loss = model(x,lengths,label_id)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_batch += 1 

            if total_batch % cf.print_per_batch == 0:
                model.eval()
                with torch.no_grad():
                    loss_train,acc_train = model.get_loss_acc(x,lengths,label_id)
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
    test_data = NewsDataset("./data/cnews_final_test.txt",cf.max_word_len,cf.max_sen_len)
    test_dataloader = DataLoader(test_data,batch_size=cf.batch_size,shuffle=True)

    # 预训练词向量矩阵
    embedding_matrix = get_pre_embedding_matrix("./data/final_vectors")
    # 模型
    model = TextHAN(cf,torch.tensor(embedding_matrix))

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
        lengths = batch["lengths"].to(device) 
        x = batch['x'].to(device) 

        with torch.no_grad():
            pred = model.get_labels(x,lengths)
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

# train()
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
            
# test()输出
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