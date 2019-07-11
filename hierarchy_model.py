# -*- encoding:utf-8 -*-
"""
Author:wangqing
Date: 20190707
实现模型的建立，模型具体细节：
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


from pytorch_pretrained_bert import BertModel
import torch.nn as nn
import torch
import torch.nn.functional as F



class Summarizer(nn.Module):
    def __init__(self,args,device):
        # 初始化模型，建立encoder和decoder
        super(Summarizer, self).__init__()
        if args.mode == "train":
            self.encoder = BertModel.from_pretrained('bert-base-cased')
        elif args.mode == "eval":
            self.encoder = BertModel()
        self.args = args
        # we choose same hiedden_size with bert embedding
        self.decoder = nn.GRU(input_size=768,hidden_size=768,num_layers=1)

        # make all of them to gpu
        self.to(device)

    def forward(self, para_dict):
        """
        构建模型，只针对一个para来
        :param input: 假设只传入一个段落，对这一个段落进行
        :return:
        """
        # Create Encoder，计算一个
        self.encoder.eval()
        para_tokens_tensor = torch.tensor([para_dict['src']]).cuda()
        para_segments_tensor = torch.tensor([para_dict['segs']]).cuda()

        self.encoded_output,_ = self.encoder(para_tokens_tensor,para_segments_tensor,output_all_encoded_layers=False)

        # send encoded_output into decoder
        self.encoded_output = torch.transpose(self.encoded_output,0,1)
        self.decoded_output,_ = self.decoder(self.encoded_output)


        return self.decoded_output