# -*- encoding:utf-8 -*-
"""
对数据进行预处理，将原始文本处理成可以送入Bert中进行训练的向量
原始文本有：src_l1.txt,tgt_l1.txt,src_l2.txt,tgt_l2.txt
其中：src_xx.txt对应文本，tgt_xx.txt对应摘要；l1表示第一层标题和文本，l2表示第二层标题和文本
"""
import re
import nltk
import numpy as np
import torch
import gc
from pytorch_pretrained_bert import BertTokenizer

class BertData():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        # 得到wordPiece中sep,cls,pad对应的id值
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']

    def preprocess_multi(self, src):
        """
        对多句文本进行处理的函数，由于是多句文本均需保留，
        所以本版本的预处理函数不对src的长度和句子数量进行限制
        :param dataset_lists:其元素为dict，dict中由src和tgt字段构成
        :return: src_idxs,segment_ids,cls_ids
        """
        """
        token2id处理思路：
        1. 将src对应的长文本进行分句加入[CLS][SEP]
        2. 将每一句进行tokenize，进而得到其在wordpiece中对应的id
        3. 最终得到的当前所有文本的
        """
        src_lists = nltk.sent_tokenize(src)
        src_tokens = []
        segment_ids = []
        cls_ids = []
        for j in range(len(src_lists)):
            sent = src_lists[j]
            sent_token = self.tokenizer.tokenize(sent)
            new_sent_token = ['[CLS]'] + sent_token + ['[SEP]']

            new_sent_token_idxs = self.tokenizer.convert_tokens_to_ids(new_sent_token)
            src_tokens += new_sent_token_idxs

            """
            segmentid,用于区分句子A和句子B，主要采用的方法是[0,1,0,1,....]这样的方法进行区分
            思路：
            1. 计算j%2的余数，如果为0，则值为0，否则就是1
            2. 根据当前token的个数分配
            """
            if (j%2) == 0:
                segment_ids += [0] * len(new_sent_token)
            else:
                segment_ids += [1] * len(new_sent_token)

        # 统计下cls_ids
        cls_ids = [i for i,t in enumerate(src_tokens) if t == self.cls_vid]

        return src_tokens, segment_ids, cls_ids

    def preprocess_single(self,sent):
        """
        对单句的文本进行预处理
        :param sent: 二级文本中的单个句子
        :return: src_idxs
        """
        src_tokens = []
        segment_ids = []

        sent_token = self.tokenizer.tokenize(sent)
        new_sent_token = ['[CLS]'] + sent_token

        src_tokens = self.tokenizer.convert_tokens_to_ids(new_sent_token)

        return src_tokens


def format_to_bert_multi(dataset_lists, save_file):
    bert_data = BertData()
    datasets = []

    for i in range(len(dataset_lists)):
        data_dict = dict()
        src = dataset_lists[i]
        if len(src) == 0:
            return None

        if bert_data.preprocess_multi(src) is not None:
            src_tokens, segment_ids, cls_ids = bert_data.preprocess_multi(src)

            b_data_dict = {'src': src_tokens, 'segs':segment_ids,'clss': cls_ids,
                           'src_txt': src}
            datasets.append(b_data_dict)

    torch.save(datasets,save_file)
    datasets = []
    gc.collect()


def format_to_bert_single(data_lists, save_file):
    """
    对二级文本中的每个单句
    :param dataset_lists:
    :param save_file:
    :return:
    """
    bert_data = BertData()
    datasets = []
    for i in range(len(data_lists)):
        sent = data_lists[i]
        src_tokens = bert_data.preprocess_single(sent)
        b_data_dict = {'src_sent':sent, 'sent_tokens': src_tokens}
        datasets.append(b_data_dict)

    torch.save(datasets, save_file)
    datasets = []
    gc.collect()


# 处理raw_text，其格式为：
# ==代表一级标题，其后面紧跟标题内容
# 不带 = 的，代表文本内容
# ===代表二级标题，其后面紧跟标题内容
# 先将此接口留出


if __name__ == '__main__':
    """
    分别对src文件和tgt文件进行处理
    """
    src_filename = "src.txt"
    tgt_filename = "tgt.txt"
    source = []
    tgt = []
    data_lists = []
    for line in open(src_filename):
        source.append(line.strip())

    for line in open(tgt_filename):
        tgt.append(line.strip())

     
    multi_file = "multi_demo.pt"
    format_to_bert_multi(source,multi_file)
    
    single_file = "single_demo.pt"
    
    format_to_bert_single(tgt, single_file)




