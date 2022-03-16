# distribution parallel script 
# python -m torch.distributed.launch --nproc_per_node=2 train_transformer_ddp.py

import torch 
from torch import device, nn 
from model import EPTransformer, Weightformer, WeightformerConfig, vgg11_bn, resnet50
from utils import parameter_dict_combine, weight_size_dict_generate, weight_dict_print 
from utils import weight_detach, weight_resize_for_model_load, cifa10_data_load
from net_parameter import vgg11_predict_layers, resnet50_predict_layers 
from tqdm import tqdm 

import argparse 
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP

# version = {'transformer', 'weightformer'} -> original transfor or weightformer 
transformer_version = 'weightformer'
network = 'resnet50'
basic_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
meta_epochs = 200 
base_epochs = 8
save_path = 'ckpt/weightformer_parameter_predictor_best.pth' 
resize_flag = True 



def main(): 
    parser = argparse.ArgumentParser(description='WeightFormer Training') 
    parser.add_argument("--local_rank", type=int, default=-1) 
    args = parser.parse_args() 
    local_rank = args.local_rank 

    if args.local_rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
    map_location = "cuda:" + str(args.local_rank)
    device = torch.device('cuda', local_rank) 


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
    if transformer_version == 'transformer':
        model = EPTransformer(N=3, padding_idx=0, size_dict=combine_weight_size_dict) 
    else:
        configuration = WeightformerConfig(combine_weight_size_dict) 
        model = Weightformer(configuration)
    model = model.to(basic_device) 

    meta_optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
    
    for epoch in range(meta_epochs): 
        for key in combine_weight_dict.keys(): 
            print(key)
            weight_m = combine_weight_dict[key].unsqueeze(0)  # (bsz, n_o, n_i*k*k)
            
            generated_weight_dict = {key: model(weight_m, key)} 
            # weight_dict_print(generated_weight_dict)
            generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device)
            detach_weight_dict = weight_detach(generated_weight_dict) 
            detach_weight_dict = detach_weight_dict.to(device) 

            base_model = ddp_base_model_initilize(detach_weight_dict, local_rank)
            target_weight_dict = train_base(base_model) 

            for t in range(base_epochs): 
                weight_m = combine_weight_dict[key].unsqueeze(0)  # (bsz, n_o, n_i*k*k)
                generated_weight_dict = {key: model(weight_m, key)}  
                generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device)
                meta_optimizer.zero_grad() 
                loss = meta_loss(generated_weight_dict, target_weight_dict, key) 
                loss.backward() 
                meta_optimizer.step() 
                print('meta loss:', loss.item()) 
            # torch.save(EPmodel.state_dict(), save_path) 



def ddp_base_model_initilize(weight_dict, local_rank): 
    if network == 'vgg11': 
        base_model = vgg11_bn() 
    else:
        base_model = resnet50() 
    
    if resize_flag == True and network == 'resnet50': 
        IN_FEATURES = base_model.fc.in_features 
        final_fc = nn.Linear(IN_FEATURES, 10) 
        base_model.fc = final_fc 
    
    base_model = base_model.to(device) 
    base_model.load_state_dict(weight_dict) 
    base_model = DDP(base_model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) 
    return base_model 



def meta_loss(generated_weight_dict, target_weight_dict, named_layer): 
    mse_loss = nn.MSELoss() 
    begin_flag = True
    for key in generated_weight_dict: 
        if key == named_layer and begin_flag == True: 
            loss = mse_loss(generated_weight_dict[key].float(), target_weight_dict[key].float()) 
        elif key == named_layer and begin_flag == False: 
            loss += mse_loss(generated_weight_dict[key].float(), target_weight_dict[key].float()) 
    return loss 



# train for total/ only fc 
def train_base(base_model, few_shot_flag=False):
    
    # dataset 
    train_loader, test_loader = cifa10_data_load(distribution=True)
    
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=0.0001) 
    loss_fn = nn.CrossEntropyLoss() 

    # train base 
    for epoch in range(base_epochs): 
        
        # validation the parameter prediction performance 

        base_model.train()
        running_loss = 0.0 
        total_part_num = len(train_loader) // 10 
        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(train_loader)) as pbar: 
            for it, (image, label) in enumerate(train_loader):
                image, label = image.to(device), label.to(device) 
                base_optimizer.zero_grad() 

                out = base_model(image) 
                loss = loss_fn(out, label) 
                loss.backward() 
                base_optimizer.step()

                running_loss += loss.item() 
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update() 
                if few_shot_flag == True: 
                    if it % total_part_num == 0: 
                        print('Train set partial: ', it // total_part_num) 
                        print('\n')
                        valid_base(base_model, test_loader, epoch)
                break 
        break
    
    return base_model.module.state_dict()



def valid_base(test_loader, model, epoch): 
    model.eval() 
    acc = .0 
    time_stamp = 0
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(test_loader)) as pbar:
        for it, (image, label) in enumerate(test_loader): 
            image, label = image.to(device), label.to(device) 
            with torch.no_grad(): 
                out = model(image) # (bsz, vob)
                predict_y = torch.max(out, dim=1)[1] #(bsz, ) 
                acc += (predict_y == label).sum().item() / predict_y.size(0)
            pbar.set_postfix(acc=acc / (it + 1))
            pbar.update() 
            time_stamp += 1 
            break 
    val_acc = acc / time_stamp 
    return val_acc 


if __name__ == '__main__': 
    main() 