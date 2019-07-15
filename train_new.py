# -*- encoding:utf-8 -*-
"""
Author: wangqing
Date: 20190715
Version:2.0
Details：
Version1.0 实现了模型的训练，在1.0 的基础上做了以下改动：
1. 对train函数进行了修改
2. 增加了test函数
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from hierarchy_model_new import Summarizer
import random
import os



def save_model(step, state_dict, model_file):
    path = os.path.join(model_file, 'model_step_%d.pt' % step)
    torch.save(state_dict,path)


def train(args):
    """
    1. 首先load data，并将data送入模型中，得到

    2. 计算loss

    3. optimizer

    :param args: 从命令行传入的参数
    :return:
    """
    lr = args.learning_rate
    iterations = args.iterations

    # 首先取出训练数据
    train_file = args.train_file
    train_data = torch.load(train_file)

    # 初始化loss,设定loss的返回时scalar
    # loss_func = nn.BCELoss(reduction='none')
    loss_func = nn.SoftMarginLoss()
    loss = 0

    """
    batch_size应该等于1，因为bert的output的shap为（batch_size, sequence_length, hidden_size)
    我们采用的不同段落进行输入，无法使用一个batch_size中多个segment进行训练。
    因此
    """
    model = Summarizer(args)
    for iter in range(iterations):
        random.shuffle(train_data)
        for i in range(len(train_data)):
            seg_dict = train_data[i]
            contrast_list = model(seg_dict)
            # # 取出contrast_list中的每一个item，进行Loss计算
            for j in range(len(contrast_list)):
                contrast_dict = contrast_list[j]
                title_index = contrast_dict['gen']
                tgt_tensor = contrast_dict['tgt']

                # cross entropy
                temp_loss = loss_func(title_index, tgt_tensor)
                loss += temp_loss

            # optimizer
            para_num = len(contrast_list)
            loss = float(loss / para_num)
            print("{}: loss {}".format(i, loss))
            optimizer = optim.Adagrad(model.parameters(), lr=lr)
            optimizer.zero_grad()
            optimizer.step()

            # checkpoint，根据iteration来计算
        print("Finish train whole dataset")
        print("loss is {}".format(loss))

        # checkpoint，每1000 iteration保存一次
        iteration = iter + 1
        if (iteration % 1000) == 0:
            state_dict = model.state_dict()
            save_model(iteration, state_dict, args.model_file)

def val(args):
    """
    validate model
    :param args:
    :return:
    """
    print("valdidation function")


def test(args):
    """
    具体做法：
    1. load checkpoint
    2. 载入测试数据，为每个段落产生标题
    3. 计算loss
    4. 将标题ID 转换成对应的文字
    但是这个版本
    :param args:
    :return:
    """
    # load checkpoint
    checkpoint = torch.load(args.checkpoint)
    model = Summarizer(args)
    model.load_state_dict(checkpoint)
    model.eval()

    # load test data
    test_data = torch.load(args.test_file)





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-batch_size", default=1, type=int)
    parser.add_argument("-iterations", default=10000, type=int)
    parser.add_argument("-train_file", default="./preprocess/multi_heading.pt")
    parser.add_argument("-test_file",default="./preprocess/multi_test.pt")
    parser.add_argument("-vocab_file", default="./preprocess/vocab.pt")
    parser.add_argument("-learning_rate", default=0.001)
    parser.add_argument("-mode", default="train")
    parser.add_argument("-model_file",default="./model_ckpt")
    parser.add_argument("-checkpoint", default="./model_ckpt/model_step_5000.ckpt")
    parser.add_argument("-predict_config",default="./bert_config.json")

    args = parser.parse_args()

    mode = args.mode
    if mode == "train":
        train(args)
    elif mode == "val":
        val(args)
    elif mode == "test":
        test(args)

