import torch 
from torch import nn 
from model import EPTransformer, weightformer, vgg11_bn, resnet50
from utils import parameter_dict_combine, weight_size_dict_generate, weight_dict_print 
from utils import weight_detach, weight_resize_for_model_load 
from train_meta_linear import train_base 
from net_parameter import vgg11_predict_layers, resnet50_predict_layers 


# version = {'transformer', 'weightformer'}
tranformer_version = 'transformer'
network = 'resnet50'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
meta_epochs = 200
base_epochs = 8
save_path = 'ckpt/transformer_parameter_predictor_best.pth' 
resize_flag = True 



def main(): 
    # 1. load parameters of base learner 
    if network == 'vgg11': 
        ckpt_path_list = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 
    if network == 'resnet50': 
        ckpt_path_list = ['ckpt/resnet50_best.pth', 'ckpt/resnet50_best.pth']
    
    weight_dict_list = [] 
    for ckpt_path in ckpt_path_list:
        weight_dict_list.append(torch.load(ckpt_path, map_location=None)) 
    combine_weight_dict, weight_size_dict = parameter_dict_combine(weight_dict_list, device, 'transformer', 'resnet50') 
    combine_weight_size_dict = weight_size_dict_generate(combine_weight_dict) 
    # weight_dict_print(combine_weight_dict) 
    print('predicted layers:', combine_weight_size_dict) 
    
    # 2. meta transformer 
    EPmodel = EPTransformer(N=3, padding_idx=0, size_dict=combine_weight_size_dict) 
    EPmodel = EPmodel.to(device) 
    if network == 'vgg11': 
        base_model = vgg11_bn() 
    else:
        base_model = resnet50() 
    
    if resize_flag == True and network == 'resnet50': 
        IN_FEATURES = base_model.fc.in_features 
        final_fc = nn.Linear(IN_FEATURES, 10) 
        base_model.fc = final_fc 

    base_model = base_model.to(device) 

    meta_optimizer = torch.optim.Adam(EPmodel.parameters(), lr=0.0001) 
    
    for epoch in range(meta_epochs): 
        for key in combine_weight_dict.keys(): 
            print(key)
            weight_m = combine_weight_dict[key].unsqueeze(0)  # (bsz, n_o, n_i*k*k)
            generated_weight_dict = {key: EPmodel(weight_m, key)} 
            
            
            generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device)
            detach_weight_dict = weight_detach(generated_weight_dict) 
            # weight_dict_print(detach_weight_dict) 

            base_model.load_state_dict(detach_weight_dict) 
            target_weight_dict = train_base(base_model) 

            for t in range(base_epochs): 
                weight_m = combine_weight_dict[key].unsqueeze(0)  # (bsz, n_o, n_i*k*k)
                generated_weight_dict = {key: EPmodel(weight_m, key)}  
                
                generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device)
                meta_optimizer.zero_grad() 
                loss = meta_loss(generated_weight_dict, target_weight_dict, key) 
                loss.backward() 
                meta_optimizer.step() 
                print('meta loss:', loss.item()) 
            # torch.save(EPmodel.state_dict(), save_path) 



def meta_loss(generated_weight_dict, target_weight_dict, named_layer): 
    mse_loss = nn.MSELoss() 
    for key in generated_weight_dict: 
        if key == named_layer: 
            loss = mse_loss(generated_weight_dict[key].float(), target_weight_dict[key].float()) 
    return loss 


if __name__ == '__main__': 
    main() 