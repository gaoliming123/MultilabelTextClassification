# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from importlib import import_module
import argparse
import pdb

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--model', type=str, required=True, help='choose a model: TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer')
parser.add_argument('--embedding', default='pre_trained', type=str, help='random or pre_trained')
parser.add_argument('--word', default=False, type=bool, help='True for word, False for char')
args = parser.parse_args()


if __name__ == '__main__':
    dataset = 'zhengzhuang'
    #dataset = 'THUCNews'  # 数据集

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    embedding = 'embedding_SougouNews.npz'
    if args.embedding == 'random':
        embedding = 'random'
    model_name = args.model  # 'TextRCNN'  # TextCNN, TextRNN, FastText, TextRCNN, TextRNN_Att, DPCNN, Transformer
    if model_name == 'FastText':
        from utils_fasttext import build_dataset, build_iterator, get_time_dif
        embedding = 'random'
    elif model_name == 'Bert':
        from utils import build_bert_dataset, build_bert_iterator, get_time_dif
    else:
        from utils import build_dataset, build_iterator, get_time_dif

    x = import_module('models.' + model_name)
    config = x.Config(dataset, embedding)
    model = x.Model(config).to(config.device)

    if model_name == 'Bert':
        start_time = time.time()
        print("Loading data...")
        train_data, dev_data, test_data = build_bert_dataset(config)
        train_iter = build_bert_iterator(train_data, config)
        dev_iter = build_bert_iterator(dev_data, config)
        test_iter = build_bert_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
    else:
        start_time = time.time()
        print("Loading data...")
        vocab, train_data, dev_data, test_data = build_dataset(config, args.word) # args.word is False
        train_iter = build_iterator(train_data, config)
        dev_iter = build_iterator(dev_data, config)
        test_iter = build_iterator(test_data, config)
        time_dif = get_time_dif(start_time)
        print("Time usage:", time_dif)
        config.n_vocab = len(vocab)

    # train
    if model_name != 'Transformer' and model_name != 'Bert':
        init_network(model)
    print(model.parameters)
    train(config, model, train_iter, dev_iter, test_iter)
