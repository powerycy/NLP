#!/usr/bin/env python
# _*_ coding: utf-8 _*_
'''
Created on 2019-05-24 17:29:27
@author: wind
'''
from tqdm import tqdm,trange
import torch
from model import Config,TextCNN
from datahelper import NewsDataset,get_pre_embedding_matrix
from torch.utils.data import Dataset,DataLoader
from torch.optim import Adam



# def evaluate(model, x_, y_):
#     model.eval()
#     for batch in tqdm(test_dataloader,"评估",total=len(test_dataloader)):
#         label_id = batch['label_id'].to(device) 
#         segment_ids = batch['segment_ids'].to(device) 
#         with torch.no_grad():
#             logits = model(segment_ids)
#         torch.floatTensor(torch.argmax(logits,1)==label_id)
           

#     """评估在某一数据上的准确率和损失"""
#     data_len = len(x_)
#     batch_eval = batch_iter(x_, y_, 128)
#     total_loss = 0.0
#     total_acc = 0.0
#     test_dataloader
#     for x_batch, y_batch in batch_eval:
#         batch_len = len(x_batch)
#         feed_dict = feed_data(x_batch, y_batch, 1.0)
#         loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
#         total_loss += loss * batch_len
#         total_acc += acc * batch_len

#     return total_loss / data_len, total_acc / data_len

def main():
    # 配置文件
    cf = Config('./data/config.yaml')
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
    model.train()
    for epoch_id in trange(epoch,desc="Epoch"):
        for step,batch in enumerate(tqdm(train_dataloader,"batch",total=len(train_dataloader))):
            label_id = batch['label_id'].to(device) 
            segment_ids = batch['segment_ids'].to(device) 

            loss = model(segment_ids,label_id)
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step%100==0:
                print("训练集",step," loss:",loss.item())
            torch.save(model.state_dict(),"./output/model.bin")

if __name__ == "__main__":
    main()
    
            
        