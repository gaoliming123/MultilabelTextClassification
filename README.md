# 中文多标签文本分裂

## 目录

```
.
├── README.md
├── models             模型文件
│   ├── Bert.py
│   ├── DPCNN.py
│   ├── FastText.py
│   ├── TextCNN.py
│   ├── TextRCNN.py
│   ├── TextRNN.py
│   ├── TextRNN_Att.py
│   └── Transformer.py
├── run.py
├── train_eval.py      训练代码
├── utils.py           工具代码
├── utils_fasttext.py 
└── zhengzhuang        数据集
    ├── data
    │   ├── class.txt
    │   ├── dev.txt
    │   ├── test.txt
    │   ├── train.txt
    │   └── vocab.pkl
    ├── log            日志
    └── saved_dict     模型保存
```

## 数据集

数据集详细介绍见[PTM](https://github.com/yao8839836/PTM)

## 运行方法

环境，使用方法等详细介绍见网址[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)。
模型中有方法`TextCNN, TextRNN, TextRNN_Att, TextRCNN, FastText, DPCNN, Transformer, Bert`，运行命令：
```
python run.py --model Bert
```

# 参考代码
[Chinese-Text-Classification-Pytorch](https://github.com/649453932/Chinese-Text-Classification-Pytorch)
