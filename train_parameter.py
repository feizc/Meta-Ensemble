import torch 
from model import ParameterProject 
from utils import parameter_dict_combine, weight_dict_print, weight_size_dict_generate, weight_resize_for_model_load
from model import vgg11_bn  

def train(): 

    # 1. download ckpt 
    ckpt_path_list = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 
    weight_dict_list = [] 
    for ckpt_path in ckpt_path_list:
        weight_dict_list.append(torch.load(ckpt_path, map_location=None)) 
    # weight_dict_print(weight_dict_list[0])
    combine_weight_dict, weight_size_dict = parameter_dict_combine(weight_dict_list)
    # weight_dict_print(combine_weight_dict)
    combine_weight_size_dict = weight_size_dict_generate(combine_weight_dict)
    

    # 2. model parameter predict 
    parameter_predict_model = ParameterProject(weight_size_dict=combine_weight_size_dict) 
    generated_weight_dict = parameter_predict_model(combine_weight_dict)
    generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0]) 
    # weight_dict_print(generated_weight_dict) 
    
    # 3. model parameter load
    vgg_model = vgg11_bn() 
    vgg_model.load_state_dict(generated_weight_dict) 
    
    # 4. training 


if __name__ == '__main__': 
    train() 
