import pickle 
from PIL import Image 
from model import vgg11_bn 
from torch import nn 
import torch 

from train_vgg import OUTPUT_DIM 


def data_plot():
    cifar10_path = 'data/cifar-10/data_batch_1' 
    with open(cifar10_path, 'rb') as f: 
        dict = pickle.load(f, encoding='bytes') 

    img = dict[b'data'][1].reshape(3, 32, 32).transpose(1,2,0)
    im = Image.fromarray(img) 
    im.show() 


def ckpt_load(ckpt_path): 
    OUTPUT_DIM = 10

    state_dict = torch.load(ckpt_path, map_location=None) 
    model = vgg11_bn() 
    model.load_state_dict(state_dict) 
    IN_FEATURES = model.classifier[-1].in_features
    final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM) 
    model.classifier[-1] = final_fc 

    #print(model) 
    #print(model.state_dict().keys())
    #for key in model.state_dict().keys():
    #    print(key, model.state_dict()[key].size()) 
    return model 


def weight_dict_print(weight_dict): 
    for key in weight_dict.keys(): 
        print(key, weight_dict[key].size())


def weight_size_dict_generate(weight_dict): 
    weight_size_dict = {} 
    for key in weight_dict.keys(): 
        weight_size_dict[key] = weight_dict[key].size() 
    return weight_size_dict 


# combine the model weights in one dict 
def parameter_dict_combine(weight_dict_list): 
    combine_weight_dict = {} 
    weight_size_dict = {}

    if len(weight_dict_list) < 1: 
        return combine_weight_dict, weight_size_dict 

    for key in weight_dict_list[0].keys(): 
        if 'classifier' in key:
            break
        weight_matrix = weight_dict_list[0][key] 
        out_dim = weight_matrix.size()[0] 
        weight_size_dict[key] = weight_matrix.size()
        weight_matrix = weight_matrix.view(out_dim, -1)
        
        for i in range(1, len(weight_dict_list)): 
            t_weight_matrix = weight_dict_list[i][key]
            t_weight_matrix = t_weight_matrix.view(out_dim, -1)
            weight_matrix = torch.cat((weight_matrix, t_weight_matrix), dim=-1) 
        
        combine_weight_dict[key] = weight_matrix 
    return combine_weight_dict, weight_size_dict 
 

 # resize the weight dict for model loading 
def weight_resize_for_model_load(weight_dict, original_weight_dict): 
    for key in original_weight_dict.keys(): 
        if key in weight_dict.keys():
            weight_dict[key] = weight_dict[key].view(original_weight_dict[key].size())
        else:
            weight_dict[key] = original_weight_dict[key]
    return weight_dict
    



if __name__ == '__main__': 
    ckpt_path = 'ckpt/vgg11_bn.pth' 
    ckpt_load(ckpt_path)