# linear baseline for parameter prediction 
import torch 
from torch import nn 
from tqdm import tqdm 
from model import ParameterProject, vgg11_bn, resnet50 
from utils import parameter_dict_combine, weight_dict_print, weight_detach,\
                    weight_size_dict_generate, weight_resize_for_model_load
from utils import cifa10_data_load

from net_parameter import vgg11_predict_layers, resnet50_predict_layers 

# network = {'vgg11', 'resnet50'}
network = 'vgg11'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
meta_epochs = 200
base_epochs = 8 
batch_size = 8
save_path = 'ckpt/linear_parameter_predictor_best.pth' 
# analyze how many ratio for training in the few shot experiments 
few_shot_flag = True 
resize_flag = True 

def main(): 
    
    # 1. download ckpt 
    if network == 'vgg11': 
        ckpt_path_list = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 
    if network == 'resnet50': 
        ckpt_path_list = ['ckpt/resnet50_best.pth', 'ckpt/resnet50_best.pth']
    
    weight_dict_list = [] 
    for ckpt_path in ckpt_path_list:
        weight_dict_list.append(torch.load(ckpt_path, map_location=None)) 
    combine_weight_dict, weight_size_dict = parameter_dict_combine(weight_dict_list, device, 'linear', network) 
    combine_weight_size_dict = weight_size_dict_generate(combine_weight_dict) 
    weight_dict_print(combine_weight_dict) 
    
    # 2. meta learner and base learner 
    parameter_predict_model = ParameterProject(weight_size_dict=combine_weight_size_dict) 
    parameter_predict_model = parameter_predict_model.to(device) 
    if network == 'vgg11': 
        base_model = vgg11_bn() 
    else:
        base_model = resnet50() 
    
    if resize_flag == True and network == 'resnet50': 
        IN_FEATURES = base_model.fc.in_features 
        final_fc = nn.Linear(IN_FEATURES, 10) 
        base_model.fc = final_fc 

    base_model = base_model.to(device)

    meta_optimizer = torch.optim.Adam(parameter_predict_model.parameters(), lr=0.0001) 

    
    for epoch in range(meta_epochs): 
        parameter_predict_model.train() 
        new_combine_weight_dict = weight_detach(combine_weight_dict)
        generated_weight_dict = parameter_predict_model(new_combine_weight_dict) 
        generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device)
        detach_weight_dict = weight_detach(generated_weight_dict) 

        base_model.load_state_dict(detach_weight_dict) 
        
        target_weight_dict = train_base(base_model) 

        
        for t in range(base_epochs): 
            
            new_combine_weight_dict = weight_detach(combine_weight_dict)
            generated_weight_dict = parameter_predict_model(new_combine_weight_dict) 
            generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device)
            loss = meta_loss(generated_weight_dict, target_weight_dict) 
            meta_optimizer.zero_grad() 
            loss.backward() 
            meta_optimizer.step() 
            print('meta loss:', loss.item()) 
            # print(parameter_predict_model.state_dict())
        
        combine_tuning(parameter_predict_model, base_model, combine_weight_dict, weight_dict_list)
        torch.save(parameter_predict_model.state_dict(), save_path) 
    


# jointly training tuning for performance boosting 
def combine_tuning(parameter_predict_model, base_model, combine_weight_dict, weight_dict_list): 
    train_loader, test_loader = cifa10_data_load() 
    meta_optimizer = torch.optim.SGD(parameter_predict_model.parameters(), lr=0.1, momentum=0.9) 

    for epoch in range(base_epochs): 
        parameter_predict_model.train() 
        new_combine_weight_dict = weight_detach(combine_weight_dict)
        generated_weight_dict = parameter_predict_model(new_combine_weight_dict) 
        generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device)
        
        base_model.load_state_dict(generated_weight_dict) 

        loss_fn = nn.CrossEntropyLoss() 
        running_loss = 0.0
        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(train_loader)) as pbar: 
            for it, (image, label) in enumerate(train_loader):
                image, label = image.to(device), label.to(device) 
                meta_optimizer.zero_grad() 

                out = base_model(image) 
                loss = loss_fn(out, label) 
                loss.backward() 
                meta_optimizer.step()

                running_loss += loss.item() 
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update() 
                break 
        break



def meta_loss(generated_weight_dict, target_weight_dict): 
    begin_flag = True 
    mse_loss = nn.MSELoss()
    for key in generated_weight_dict: 
        if network == 'vgg11' and key not in vgg11_predict_layers: 
            continue 
        if network == 'resnet50' and key not in resnet50_predict_layers: 
            continue
        if begin_flag == True: 
            loss = mse_loss(generated_weight_dict[key].float(), target_weight_dict[key].float()) 
            begin_flag = False 
        else:
            loss += mse_loss(generated_weight_dict[key].float(), target_weight_dict[key].float())  
    return loss



# train for total/ only fc 
def train_base(base_model):
    
    # dataset 
    train_loader, test_loader = cifa10_data_load()
    
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=0.0001) 
    loss_fn = nn.CrossEntropyLoss() 

    # train base 
    for epoch in range(base_epochs): 
        
        # validation the parameter prediction performance 

        base_model.train()
        running_loss = 0.0 
        total_part_num = len(train_loader) // 10 
        val_acc_list = []
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
                        val_acc = simple_valid_base(base_model, test_loader, epoch) 
                        val_acc_list.append(val_acc)
                break 
        print('part val acc:', val_acc_list)
        break
    return base_model.state_dict()



def valid_base(base_model, test_loader, epoch=0):
    acc = .0 
    time_stamp = 0 
    base_model.eval() 
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(test_loader)) as pbar:
        for it, (image, label) in enumerate(test_loader): 
            image, label = image.to(device), label.to(device) 
            with torch.no_grad(): 
                out = base_model(image) # (bsz, vob)
                predict_y = torch.max(out, dim=1)[1] #(bsz, ) 
                acc += (predict_y == label).sum().item() / predict_y.size(0)
            pbar.set_postfix(acc=acc / (it + 1))
            pbar.update() 
            time_stamp += 1 
            break 
        val_acc = acc / time_stamp
    return val_acc 



def simple_valid_base(base_model, test_loader, epoch=0):
    acc = .0 
    time_stamp = 0 
    base_model.eval() 
    for it, (image, label) in enumerate(test_loader): 
        image, label = image.to(device), label.to(device) 
        with torch.no_grad(): 
            out = base_model(image) # (bsz, vob)
            predict_y = torch.max(out, dim=1)[1] #(bsz, ) 
            acc += (predict_y == label).sum().item() / predict_y.size(0)
            
        time_stamp += 1 
        break 
    val_acc = acc / time_stamp
    return val_acc 



if __name__ == '__main__': 
    main() 
