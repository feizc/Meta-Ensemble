import torch 
from model import ParameterProject 
from utils import parameter_dict_combine, weight_dict_print, weight_size_dict_generate


def train(): 

    ckpt_path_list = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 
    weight_dict_list = [] 

    for ckpt_path in ckpt_path_list:
        weight_dict_list.append(torch.load(ckpt_path, map_location=None)) 
    
    # weight_dict_print(weight_dict_list[0])
    combine_weight_dict, weight_size_dict = parameter_dict_combine(weight_dict_list)
    # weight_dict_print(combine_weight_dict)
    combine_weight_size_dict = weight_size_dict_generate(combine_weight_dict)
    
    model = ParameterProject(weight_size_dict=combine_weight_size_dict) 
    loss = model(combine_weight_dict)
    print(loss)

if __name__ == '__main__': 
    train() 
