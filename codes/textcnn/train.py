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
from model import Config,TextCNN
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
    for batch in tqdm(test_dataloader,"评估",total=len(test_dataloader)):
        label_id = batch['label_id'].squeeze(1).to(device) 
        segment_ids = batch['segment_ids'].to(device) 
        with torch.no_grad():
            loss,acc = model.get_loss_acc(segment_ids,label_id)
        
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
    model = TextCNN(cf,torch.tensor(embedding_matrix))
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
        for step,batch in enumerate(tqdm(train_dataloader,"batch",total=len(train_dataloader))):
            
            label_id = batch['label_id'].squeeze(1).to(device) 
            segment_ids = batch['segment_ids'].to(device) 

            loss = model(segment_ids,label_id)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            total_batch += 1 

            if total_batch % cf.print_per_batch == 0:
                model.eval()
                with torch.no_grad():
                    loss_train,acc_train = model.get_loss_acc(segment_ids,label_id)
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
    model = TextCNN(cf,torch.tensor(embedding_matrix))

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
    for step,batch in enumerate(tqdm(test_dataloader,"batch",total=len(test_dataloader))):
        
        label_id = batch['label_id'].squeeze(1).to(device) 
        segment_ids = batch['segment_ids'].to(device) 
        with torch.no_grad():
            pred = model.get_labels(segment_ids)
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
            
# test()输出

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