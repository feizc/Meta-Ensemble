import torch 
from torch import nn 
from model import EPTransformer, vgg11_bn 
from utils import parameter_dict_combine, weight_size_dict_generate, weight_dict_print 
from utils import weight_detach, weight_resize_for_model_load 
from train_meta_linear import train_base 



device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
meta_epochs = 200
base_epochs = 8
save_path = 'ckpt/transformer_parameter_predictor_best.pth' 

def train(): 
    # 1. load parameters of base learner 
    ckpt_path_list = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 
    weight_dict_list = [] 
    for ckpt_path in ckpt_path_list:
        weight_dict_list.append(torch.load(ckpt_path, map_location=None)) 
    combine_weight_dict, weight_size_dict = parameter_dict_combine(weight_dict_list, device, 'transformer') 
    combine_weight_size_dict = weight_size_dict_generate(combine_weight_dict) 
    # weight_dict_print(combine_weight_dict) 

    # 2. meta transformer 
    EPmodel = EPTransformer(N=3, padding_idx=0) 
    EPmodel = EPmodel.to(device) 
    base_model = vgg11_bn() 
    base_model = base_model.to(device) 

    meta_optimizer = torch.optim.Adam(EPmodel.parameters(), lr=0.0001) 
    
    for epoch in range(meta_epochs): 
        for key in combine_weight_dict.keys(): 
            
            weight_m = combine_weight_dict[key].unsqueeze(0)  # (bsz, n_o, n_i*k*k)
            generated_weight_dict = {'features.0.weight': EPmodel(weight_m)} 
            generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device)
            detach_weight_dict = weight_detach(generated_weight_dict) 
            # weight_dict_print(detach_weight_dict) 

            base_model.load_state_dict(detach_weight_dict) 
            target_weight_dict = train_base(base_model) 

            for t in range(base_epochs): 
                weight_m = combine_weight_dict[key].unsqueeze(0)  # (bsz, n_o, n_i*k*k)
                generated_weight_dict = {'features.0.weight': EPmodel(weight_m)} 
                
                generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device)
                meta_optimizer.zero_grad() 
                loss = meta_loss(generated_weight_dict, target_weight_dict) 
                loss.backward() 
                meta_optimizer.step() 
                print('meta loss:', loss.item()) 
            # torch.save(EPmodel.state_dict(), save_path) 



def meta_loss(generated_weight_dict, target_weight_dict): 
    mse_loss = nn.MSELoss() 
    for key in generated_weight_dict: 
        if 'features.0.weight' in key: 
            loss = mse_loss(generated_weight_dict[key].float(), target_weight_dict[key].float()) 
    return loss 




if __name__ == '__main__': 
    train() 