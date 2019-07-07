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
3. softmax
P = softmax(wx+b)
根据vocab_size，可以得到w的shape为（vocab_size,hidden_size)
x的shape为（seq_len,batch,hidden_size)
先对x进行reshape成（seq_len,hidden_size)
这块有疑问，softmax是需要指定维度的，如果指定维度，则计算出的distribution不对
因此这块需要重新考虑，向量维度如何变化
"""


from pytorch_pretrained_bert import BertModel
import torch.nn as nn



class Summarizer(nn.Module):
    def __init__(self,args):
        # 初始化模型，建立encoder和decoder
        super(Summarizer, self).__init__()
        self.encoder = BertModel.from_pretrained('bert-base-cased')
        self.args = args
        # we choose same hiedden_size with bert embedding
        self.decoder = nn.GRU(input_size=768,hidden_size=768,num_layers=1)

    def forward(self, *input):
        # 模型的具体实现方式