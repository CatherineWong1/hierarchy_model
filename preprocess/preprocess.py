# -*- encoding:utf-8 -*-
"""
Author: wangqing
Date: 20190704
实现的功能：
1. 对src.txt和tgt.txt进行预处理，得到处理好的数据集，将数据集处理成Json的格式，但保存为.pt文件。
由于需要考虑在训练时，选一个batch_size的数据，因此数据集格式如下：
[{ "segment":
  [{"src_txt":原文段落1, "tgt_txt":标题1, src:原文对应的wordPiece id, segs:原文对应的句子分隔, tgt:标题对应的wordPiece id},
   {"src_txt":原文段落1.1, "tgt_txt":标题1.1, src_tokens:原文对应的wordPiece id, src_cls:原文对应的句子分隔, tgt_tokens:标题对应的wordPiece id}]
}
2. 对src.txt和tgt.txt中的所有单词，生成一个无重复的词汇表

"""

import nltk
import torch
import gc
import argparse
from pytorch_pretrained_bert import BertTokenizer


class BertData():
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        # 得到wordPiece中sep,cls,pad对应的id值
        self.sep_vid = self.tokenizer.vocab['[SEP]']
        self.cls_vid = self.tokenizer.vocab['[CLS]']
        self.pad_vid = self.tokenizer.vocab['[PAD]']
        self.vocab = []

    def preprocess(self, src, tgt):
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
        val_token = []
        for j in range(len(src_lists)):
            sent = src_lists[j]
            sent_token = self.tokenizer.tokenize(sent)
            if len(sent_token) > 510:
                sent_token = sent_token[:510]
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

        # 截断整个段落的长度
        if len(src_tokens) > 512:
            src_tokens = src_tokens[:512]
            segment_ids = segment_ids[:512]
        # create vocabulary
        self.vocab = self.create_vocabulary(src_lists, tgt)
        

        return src_tokens, segment_ids

    def create_vocabulary(self, sent_lists, tgt):
        for sent in sent_lists:
            word_lists = nltk.word_tokenize(sent)
            for word in word_lists:
                try:
                    index = self.vocab.index(word)
                except Exception as e:
                    self.vocab.append(word)

        tgt_word = nltk.word_tokenize(tgt)
        for t_word in tgt_word:
            try:
                index = self.vocab.index(t_word)
            except Exception as e:
                self.vocab.append(t_word)
        return self.vocab

    def convert_tgt_to_ids(self, tgt):
        tgt_tokens = []
        for i in range(len(tgt)):
            token = tgt[i]
            index = self.vocab.index(token)
            tgt_tokens.append(index)

        return tgt_tokens


def format_to_bert(args):
    """
    将raw_lists转换成所需的数据格式
    :param raw_lists: 整合src.txt和tgt.txt后的数据列表
    :param save_file: 需要保存的数据集名称
    :return:
    """
    bert_data = BertData()
    vocabulary = []

    src_file = args.src_file
    tgt_file = args.tgt_file
    raw_lists = load_data(src_file, tgt_file)

    for i in range(len(raw_lists)):
        print("This is {} raw data".format(i))
        item_list = raw_lists[i]['segment']
        for item in item_list:
            src = item['src_txt']
            tgt = item['tgt_txt']
            src_tokens, segment_ids = bert_data.preprocess(src,tgt)
            item['src'] = src_tokens
            item['segs'] = segment_ids

    vocabulary = bert_data.vocab
    vocabulary.sort()
    torch.save(vocabulary, args.vocab_file)

    # 处理tgt:
    for i in range(len(raw_lists)):
        item_list = raw_lists[i]['segment']
        print('{}: item_list_len:{}'.format(i,len(item_list)))
        for item in item_list:
            src = item['src_txt']
            tgt = item['tgt_txt']
            tgt_token = nltk.word_tokenize(tgt)
            tgt_tokens = bert_data.convert_tgt_to_ids(tgt_token)
            item['tgt'] = tgt_tokens

    torch.save(raw_lists, args.dataset_file)
    gc.collect()


def load_data(src_file,tgt_file):
    """
    将raw_txt整合成后续可处理的列表嵌套字典的格式
    :param src_file: src.txt
    :param tgt_file: tgt.txt
    :return: raw_lists
    """
    raw_lists = []
    item_flag = 0
    tgt_f = open(tgt_file, 'r', errors='ignore')

    with open(src_file,'r',errors='ignore') as src_f:
        item_lists = []
        for line in src_f.readlines():
            if line.count('\n') == len(line):
                if item_flag == 0:
                    item_flag = 1
                    item_dict = {"segment": item_lists}
                    raw_lists.append(item_dict)
                    item_lists = []
                    continue
                else:
                    continue
            else:
                item_flag = 0
                raw_src = line.strip()
                try:
                    raw_tgt = next(tgt_f)
                    if raw_tgt.count('\n') == len(raw_tgt):
                        raw_tgt = next(tgt_f)
                        if raw_tgt.count('\n') == len(raw_tgt):
                            raw_tgt = next(tgt_f)
                    raw_tgt = raw_tgt.strip()
                    raw_dict = {"src_txt":raw_src, "tgt_txt":raw_tgt}
                    item_lists.append(raw_dict)
                except Exception as e:
                    break
 
    return raw_lists


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_file",default='multi_heading.pt')
    parser.add_argument("-vocab_file", default='vocab.pt')
    parser.add_argument("-src_file",default='src.txt')
    parser.add_argument("-tgt_file", default='tgt.txt')

    args = parser.parse_args()

    format_to_bert(args)
