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

def train(args):
    """
    1. 首先load data，并将data送入Bert中，并生成向量
    2. 将生成的向量送入单层LSTM中，后街softmax得到vocab的prediction
    3. 计算loss
    :param args: 从命令行传入的参数
    :return:
    """
    # 初始化参数
    batch_size = args.batch_size
    iterations = args.iterations


    # 首先取出训练数据
    train_file = args.train_file
    train_data = torch.load(train_file)




def val(args):
    """
    validate model
    :param args:
    :return:
    """
    print("valdidation function")



def predict(args):
    """
    generate heading from model
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
    parser.add_argument("-mode",default="train")

    args = parser.parse_args()

    mode = args.mode
    if mode == "train":
        train(args)
    elif (mode == "val"):
        val(args)
    elif (mode == "predict"):
        predict(args)
