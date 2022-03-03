#!/usr/bin/env python
# encoding: utf-8
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertConfig, BertModel
from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from transformers import XLNetTokenizer, XLNetModel, XLNetConfig

import pdb

class Config(object):
    """配置参数"""
    def __init__(self, dataset, embedding):
        self.model_name = 'Bert'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.num_classes = len(self.class_list)                         # 类别数
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_epochs = 100                                           # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.learning_rate = 1e-4                                       # 学习率
        self.feature_layers = 5
        self.dropout = 0.5                                              # 随机失活
        self.max_length = 32

def get_bert(bert_name):
    if 'roberta' in bert_name:
        print('load roberta-base')
        model_config = RobertaConfig.from_pretrained('roberta-base')
        model_config.output_hidden_states = True
        bert = RobertaModel.from_pretrained('roberta-base', config=model_config)
    elif 'xlnet' in bert_name:
        print('load xlnet-base-cased')
        model_config = XLNetConfig.from_pretrained('xlnet-base-cased')
        model_config.output_hidden_states = True
        bert = XLNetModel.from_pretrained('xlnet-base-cased', config=model_config)
    else:
        print('load bert-base-chinese')
        model_config = BertConfig.from_pretrained('bert-base-chinese')
        model_config.output_hidden_states = True
        bert = BertModel.from_pretrained('bert-base-chinese', config=model_config)
    return bert

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.device = config.device
        self.feature_layers = config.feature_layers
        self.bert_name, self.bert = config.model_name, get_bert(config.model_name).to(config.device)
        self.l0 = nn.Linear(self.feature_layers*self.bert.config.hidden_size, config.num_classes)
        self.tokenizer = self.get_tokenizer()
        self.drop_out = nn.Dropout(config.dropout)

    def get_tokenizer(self):
        tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', do_lower_case=True)
        return tokenizer

    def forward(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        input_ids = inputs['input_ids'].to(self.device)
        token_type_ids = inputs['token_type_ids'].to(self.device)
        attention_mask = inputs['attention_mask'].to(self.device)
        outs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)[-1]
        out = torch.cat([outs[-i][:, 0] for i in range(1, self.feature_layers+1)], dim=-1)
        out = self.drop_out(out)
        logits = self.l0(out)
        return logits

