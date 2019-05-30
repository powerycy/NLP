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
from pytorch_pretrained_bert import BertForSequenceClassification, BertAdam


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
    # 模型
    if os.path.isfile("./bert/bert-base-chinese.tar.gz"):
        model = BertForSequenceClassification.from_pretrained('./bert/bert-base-chinese.tar.gz',num_labels=cf.num_labels)
    else:
        model = BertForSequenceClassification.from_pretrained('bert-base-chinese',num_labels=cf.num_labels)
    
    # 优化器用adam
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
    require_improvement = 1500  # 如果超过1000轮未提升，提前结束训练

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
    train()
    # test()

# train()输出
#            
# test()输出