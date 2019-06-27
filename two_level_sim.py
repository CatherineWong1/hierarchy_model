# -*- encoding:utf-8 -*-

from pytorch_pretrained_bert import BertModel
import torch


def load_data(file1,file2):
    sent_high_lists = torch.load(file1)
    sent_low_lists = torch.load(file2)

    high_text = dict()
    high_text = sent_high_lists[0]

    return high_text, sent_low_lists

def cal_sim(high_text, sent_low_lists):
    rank_list = []
    print("*************fcuntion cal_sim**********")
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
   
    high_tokens_tensor = torch.tensor([high_text['src']])
    high_segments_tensor = torch.tensor([high_text['segs']])

    # make tensors to GPU
    high_tokens_tensor = high_tokens_tensor.to('cuda')
    high_segments_tensor = high_segments_tensor.to('cuda')

    model.to('cuda')

    with torch.no_grad():
        high_encoded_layers, _ = model(high_tokens_tensor, high_segments_tensor)

    high_output_tensor = high_encoded_layers[1]
    print(high_output_tensor.shape)

    high_length = len(high_text['src'])

    sim_res = torch.tensor([0])
    for i in range(len(sent_low_lists)):
       low_item = sent_low_lists[i]
       low_sent_tokens = low_item['sent_tokens']
       low_length = len(low_sent_tokens)
       if low_length < high_length:
           padding = (high_length-low_length)*[0]
           low_sent_tokens += padding
           low_sent_tokens_tensor = torch.tensor([low_sent_tokens])
           low_sent_tokens_tensor = low_sent_tokens_tensor.to('cuda')
           with torch.no_grad():
               low_encoder_layers,_ = model(low_sent_tokens_tensor)
       
           low_sent_output = low_encoder_layers[1]
           print(low_sent_output.shape)
           print("**************cal similarity************")
           cos = torch.nn.CosineSimilarity()
           if i == 0:
               sim_res = cos(torch.reshape(high_output_tensor,(1,-1)),torch.reshape(low_sent_output,(1,-1)))
           else:
               temp_res = cos(torch.reshape(high_output_tensor,(1,-1)),torch.reshape(low_sent_output,(1,-1)))
               sim_res = torch.cat((sim_res,temp_res))

    print("***********Process tensor rank*************")
    sort_res,indicies = torch.sort(sim_res,descending=True)
    index_res = indicies.cpu().numpy().tolist()
    print(index_res)
    print(sort_res)
    print("**********Create dict to show the rank result***********")
    for j in range(len(index_res)):
        item = index_res[j]
        res_dict = {"score":sort_res[j],"txt":sent_low_lists[item]['src_sent']}
        rank_list.append(res_dict)
           
    print(rank_list)

if __name__ == '__main__':
    file1 = "./preprocess/multi_demo.pt"
    file2 = "./preprocess/single_demo.pt"

    high_text, sent_low_lists = load_data(file1, file2)
    
    cal_sim(high_text, sent_low_lists)
