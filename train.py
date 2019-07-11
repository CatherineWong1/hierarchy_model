# -*- encoding:utf-8 -*-
"""
Author: wangqing
Date: 20190705
实现模型的训练：
需要人为设置的参数:
1. 数据：训练数据：multi_heading.pt, 词典数据：vocab.pt
2. batch_size,iteration, lstm unit，mode
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hierarchy_model import Summarizer
import random


def get_maxlen(para_list):
    """
    获得一个segment中最长para的seq_len
    :param para_list: 一个segment下所有的段落
    :return: max_len
    """
    max_len = 0
    for i in range(len(para_list)):
        para_dict = para_list[i]
        if i == 0:
            max_len = len(para_dict['src'])
        else:
            src_len = len(para_dict['src'])
            if src_len > max_len:
                max_len = src_len

    return max_len


def padding_tensor(src_len,max_len):
    """
    创建需要padding的tensor
    :param src_len: 当前段落的seq_len
    :param max_len: 最长段落的seq_len
    :return: zero_tensor
    """
    padding_len = max_len - src_len
    zero_tensor = torch.tensor((), dtype=torch.float32)
    batch_size=1
    hidden_size=768
    zero_tensor = zero_tensor.new_zeros((padding_len, batch_size,hidden_size))
    return zero_tensor


def train(args):
    """
    1. 首先load data，并将data送入Bert中，并生成向量
    2. 将生成的向量送入单层LSTM中，后街softmax得到vocab的prediction
    做法：
    seq_len对齐后，进行相加，然后使用torch.sum(x,dim=0)可以变成shape(batch_size,hidden_size)
    P = softmax(wx+b)
    根据vocab_size，可以得到W的shape为(vocab_size, hidden_size)
    x的shape应为（batch,hidden_size)，初始化使用正态分布
    b的shape为[vocab_size]，初始化使用正态分布

    3. 计算loss
     y = torch.nn.functional.linear(input=x,weight = W，bias=b)
    y的shape为(batch_size,vocab_size)
    res = torch.nn.funcational.linear(y)
    vocab_distribution = torch.softmax(res) # shape为(batch_size, vocab_size）
    # topk返回的是output,output_indicies
    title = torch.topk(vocab_distribution,5)
    将gold summary的文本在vocab中找到相关id，进行cross entropy

    4. optimizer
    简单写

    5. 一次全数据的训练相当于一个epoch，N次iteration
    一般一次iteration就是一次数据集为batch_size大小的训练
    因此需要多次迭代


    :param args: 从命令行传入的参数
    :return:
    """
    # 初始化参数
    batch_size = args.batch_size
    iterations = args.iterations
    hidden_size = 768
    lr = args.learning_rate

    # 首先取出训练数据
    train_file = args.train_file
    train_data = torch.load(train_file)
    vocab_file = args.vocab_file
    vocab = torch.load(vocab_file)
    vocab_size = len(vocab)

    # 初始化参数
    w = torch.randn((batch_size, hidden_size), requires_grad=True)
    b = torch.randn((vocab_size), requires_grad=True)
    # 初始化loss,设定loss的返回时scalar
    loss_func = nn.BCELoss(reduction='none')
    loss = 0


    """
    batch_size应该等于1，因为bert的output的shap为（batch_size, sequence_length, hidden_size)
    我们采用的不同段落进行输入，无法使用一个batch_size中多个segment进行训练。
    因此
    """
    device = 'cuda'
    model = Summarizer(args,device)
    random.shuffle(train_data)
    for i in range(len(train_data)):
        # 取出一个段落的标题
        seg_dict = train_data[i]
        para_list = seg_dict['segment']
        # 针对一个segment计算softmax和loss
        para_num = len(para_list)
        max_len = get_maxlen(para_list)

        # 对齐所有段落
        padding_list = []
        for j in range(para_num):
            para_dict = para_list[j]
            src_len = len(para_dict['src'])
            para_output = model(para_dict)
            if src_len < max_len:
                zero_tensor = padding_tensor(src_len, max_len)
                para_tensor = torch.cat((para_output, zero_tensor), 0)
                padding_list.append(para_tensor)
            else:
                padding_list.append(para_output)

        # 将当前段落的output和其他所有段落tensor的均值加到一起,并sum成shape为(batch_size,hidden_size)
        avg_list = [0] * para_num
        for m in range(para_num):
            for n in range(para_num):
                if n == m:
                    continue
                else:
                    avg_list[m] += padding_list[n]

            temp_avg = avg_list[m] / 4
            avg_list[m] = torch.sum(temp_avg,dim=0)

        # 计算loss，进行gradient
        for j in range(para_num):
            para_dict = para_list[j]
            # 计算softmax
            input = avg_list[j]
            linear_res = F.linear(input, w, b)
            softmax_res = F.softmax(linear_res)

            # 根据softmax选择top 5个单词，作为标题
            top_res = torch.topk(softmax_res, 5)
            title_index = top_res[1]

            # cross entropy
            tgt = para_dict['tgt']
            tgt_tensor = torch.tensor(tgt,dtype=torch.float64)
            loss_func = (title_index,tgt_tensor)
            loss += loss_func

        # optimizer
        loss = float(loss/para_num)
        optimizer = optim.Adagrad([w,b],lr=lr)
        optimizer.step(loss)

        # checkpoint，根据iteration来计算


def val(args):
    """
    validate model
    :param args:
    :return:
    """
    print("valdidation function")



def predict(args):
    """
    具体做法：
    1. load checkpoint
    2. 将
    :param args:
    :return:
    """
    print("predict function")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size",default=64, type=int)
    parser.add_argument("-iterations", default=10000,type=int)
    parser.add_argument("-lstm_unit", default=128, type=int)
    parser.add_argument("-train_file", default="multi_heading.pt")
    parser.add_argument("-vocab_file",default="vocab.pt")
    parser.add_argument("-learning_rate",default=0.001)
    parser.add_argument("-mode",default="train")

    args = parser.parse_args()

    mode = args.mode
    if mode == "train":
        train(args)
    elif mode == "val":
        val(args)
    elif mode == "predict":
        predict(args)
