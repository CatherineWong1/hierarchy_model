# -*- encoding:utf-8 -*-
"""
Author:wangqing
Date: 20190715
Version: 2.0
Details：
Version1.0 仅实现了单个Model的
针对一个Para的具体细节：
1. Encoder部分
由于各个段落的长度不一致，因此分别将各个段落送入BertModel中
经过BertModel得到embedding
在这个过程中，注意到有一个函数，model.train()或者model.eval()
这个两个参数仅对模型中有dropout时有影响。
Encoder出来后，得到的output为;encoder_layer,pooled_output,
encoder_layer中的结果为我们所需的hidden_state
2. Decoder
input_size = hidden_size =encoder_layer.hidden_size
采用一层GRU
encoder中的hidden_state即为input,decoder中的hidden_state=None
它不需要hidden_state
由于output的shape为(seq_len,batch_size,hidden_size)

"""

from pytorch_pretrained_bert import BertModel, BertConfig
import torch.nn as nn
import torch
import torch.nn.functional as F


class Summarizer(nn.Module):
    def __init__(self, args):
        # 初始化模型，建立encoder和decoder
        super(Summarizer, self).__init__()
        if args.mode == "train":
            self.encoder = BertModel.from_pretrained('bert-base-cased', cache_dir="./temp")
        elif args.mode == "test":
            config = BertConfig.from_json_file(args.predict_config)
            self.encoder = BertModel(config)
        self.args = args
        # we choose same hiedden_size with bert embedding
        self.decoder = nn.GRU(input_size=768, hidden_size=768, num_layers=1)

        # 初始化参数
        self.vocab_size = len(torch.load(args.vocab_file))
        self.hidden_size = 768
        w = torch.randn((self.vocab_size, self.hidden_size), requires_grad=True)
        b = torch.randn((self.vocab_size), requires_grad=True)

        # make all of them to gpu
        # self.to(device)

    def forward(self, seg_dict):
        """
        构建模型，针对一个segment
        :param seg_dict: 针对一个segment中的
        :return: 一个list，每个Item是一个字典，字典中的字段是生成的title,和原title
        """
        # 循环取出一个segment中的每一个para，对齐数据长度
        para_list = para_list = seg_dict['segment']
        para_num = len(para_list)
        max_len = get_maxlen(para_list)
        padding_list = []
        for j in range(para_num):
            print("This is {} para".format(j))
            para_dict = para_list[j]
            src_len = len(para_dict['src'])
            para_output = self.single_para_model(para_dict)
            if src_len < max_len:
                zero_tensor = padding_tensor(src_len, max_len)
                para_tensor = torch.cat((para_output, zero_tensor), 0)
                padding_list.append(para_tensor)
            else:
                padding_list.append(para_output)

        # 将当前段落的output和其他所有段落tensor的均值加到一起,并sum成shape为(batch_size,hidden_size)
        avg_list = []
        for m in range(para_num):
            temp_avg = torch.empty((max_len, 1, self.hidden_size))
            for n in range(para_num):
                if n == m:
                    continue
                else:
                    temp_avg += padding_list[n]

            temp_avg = temp_avg / 4
            avg = torch.sum(temp_avg, dim=0)
            avg_list.append(avg)

        # 对avg_list中的每一项进行标题的预测，并和原标题组成dict
        contrast_list = []
        for j in range(para_num):
            contrast_dict = dict()
            para_dict = para_list[j]
            # 计算softmax
            input_avg = avg_list[j]
            linear_res = F.linear(input_avg, self.w, self.b)
            softmax_res = F.softmax(linear_res, dim=1)

            # 根据softmax选择top 10个单词，作为标题，并组成数据
            top_res = torch.topk(softmax_res, 10)
            title_index = top_res[1].reshape((10))
            contrast_dict['gen'] = title_index

            goal_tgt = para_dict['tgt']
            pad_goal_tgt = align_tgt(goal_tgt)
            pad_goal_tgt_tensor = torch.tensor(pad_goal_tgt, dtype=torch.float32)
            contrast_dict['tgt'] = pad_goal_tgt_tensor

            contrast_list.append(contrast_dict)

        return contrast_list


    def single_para_model(self,para_dict):
        para_tokens_tensor = torch.tensor([para_dict['src']])
        para_segments_tensor = torch.tensor([para_dict['segs']])
        print(para_tokens_tensor)
        print(para_segments_tensor)

        self.encoded_output, _ = self.encoder(para_tokens_tensor, para_segments_tensor, output_all_encoded_layers=False)

        # send encoded_output into decoder
        self.encoded_output = torch.transpose(self.encoded_output, 0, 1)
        self.decoded_output, _ = self.decoder(self.encoded_output)

        return self.decoded_output


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


def padding_tensor(src_len, max_len):
    """
    创建需要padding的tensor
    :param src_len: 当前段落的seq_len
    :param max_len: 最长段落的seq_len
    :return: zero_tensor
    """
    padding_len = max_len - src_len
    zero_tensor = torch.tensor((), dtype=torch.float32)
    batch_size = 1
    hidden_size = 768
    zero_tensor = zero_tensor.new_zeros((padding_len, batch_size, hidden_size))
    return zero_tensor


def align_tgt(tgt):
    """
    对齐gold summary和generate title
    小于10则补齐，大于10则截断
    :param tgt: gold summary在vocabulary中的Index
    :return:
    """
    if len(tgt) < 10:
        pad_len = 10 - len(tgt)
        tgt += [0] * pad_len
    else:
        tgt = tgt[:10]

    return tgt
