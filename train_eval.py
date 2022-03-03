# coding: UTF-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
from utils import get_time_dif
from tensorboardX import SummaryWriter
import pdb
from utils import precision_n
from utils import recall_n

from transformers import AdamW
from apex import amp

# 权重初始化，默认xavier
def init_network(model, method='xavier', exclude='embedding', seed=123):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                if method == 'xavier':
                    nn.init.xavier_normal_(w)
                elif method == 'kaiming':
                    nn.init.kaiming_normal_(w)
                else:
                    nn.init.normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)
            else:
                pass


def train(config, model, train_iter, dev_iter, test_iter):
    start_time = time.time()
    model.train()
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    loss_fn = nn.BCEWithLogitsLoss()

    # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    total_batch = 0  # 记录进行到多少batch
    dev_best_loss = float('inf')
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    writer = SummaryWriter(logdir=config.log_path + '/' + time.strftime('%m-%d_%H.%M', time.localtime()))
    for epoch in range(config.num_epochs):
        print('Epoch [{}/{}]'.format(epoch + 1, config.num_epochs))
        # scheduler.step() # 学习率衰减
        for i, (trains, labels) in enumerate(train_iter):
            outputs = model(list(trains))
            model.zero_grad()
            if hasattr(model, 'bert_name'):
                labels = labels.to(config.device)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()
            if total_batch % 100 == 0:
                # 每多少轮输出在训练集和验证集上的效果
                true = labels.data.cpu()
                train_acc = precision_n(outputs.detach().cpu(), labels.detach().cpu(), n=20)
                dev_acc, dev_loss = evaluate(config, model, dev_iter)
                if dev_loss < dev_best_loss:
                    dev_best_loss = dev_loss
                    torch.save(model.state_dict(), config.save_path)
                    improve = '*'
                    last_improve = total_batch
                else:
                    improve = ''
                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}'
                print(train_acc)
                print(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                writer.add_scalar("loss/train", loss.item(), total_batch)
                writer.add_scalar("loss/dev", dev_loss, total_batch)
                writer.add_scalar("acc/train", train_acc, total_batch)
                writer.add_scalar("acc/dev", dev_acc, total_batch)
                model.train()
            total_batch += 1
            if total_batch - last_improve > config.require_improvement:
                # 验证集loss超过1000batch没下降，结束训练
                print("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break
    writer.close()
    test(config, model, test_iter)


def test(config, model, test_iter):
    # test
    model.load_state_dict(torch.load(config.save_path))
    model.eval()
    start_time = time.time()
    p5, p10, p20, r5, r10, r20, test_loss = evaluate(config, model, test_iter, test=True)
    msg = 'Test Loss: {0:>5.2},  Test Precision@5: {1:>6.2%}, Test Precision@10: {2:>6.2%}, Test Precision@20: {3:>6.2%}, Test Recall@5: {4:>6.2%}, Test Recall@5: {5:>6.2%}, Test Recall@10: {6:>6.2%}, Test Recall@20: {7:>6.2%}, '
    print(msg.format(test_loss, p5, p10, p20, r5, r10, r20))
    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)


def evaluate(config, model, data_iter, test=False):
    loss_fn = nn.BCEWithLogitsLoss()
    model.eval()
    loss_total = 0
    predict_all = []
    labels_all = []
    with torch.no_grad():
        for texts, labels in data_iter:
            outputs = model(texts)
            if hasattr(model, 'bert_name'):
                labels = labels.to(config.device)
            loss = loss_fn(outputs, labels)
            loss_total += loss
            labels = labels.data.cpu()
            predict = outputs.data.cpu()
            labels_all.append(labels)
            predict_all.append(predict)
    labels_all = torch.cat(labels_all, dim=0)
    predict_all = torch.cat(predict_all, dim=0)
    acc = precision_n(predict_all, labels_all, n=20)
    if test:
        p5 = precision_n(predict_all, labels_all, n=5)
        p10 = precision_n(predict_all, labels_all, n=10)
        p20 = precision_n(predict_all, labels_all, n=20)
        r5 = recall_n(predict_all, labels_all, n=5)
        r10 = recall_n(predict_all, labels_all, n=10)
        r20 = recall_n(predict_all, labels_all, n=20)
        return p5, p10, p20, r5, r10, r20, loss_total / len(data_iter)
    return acc, loss_total / len(data_iter)
