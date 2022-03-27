# script for multiple checkpoints evaluation and report the statics results 
from torch import nn 
import torch  
from tqdm import tqdm 
import numpy as np 

from model import vgg11_bn, resnet50
from train_vgg import OUTPUT_DIM


from utils import cifa10_data_load, cifa100_data_load, imagenet_data_load 

# {'CIFAR-10', 'CIFAR-100', 'ImageNet'}
dataset_name = 'CIFAR-10'
# {'VggNet-11', 'ResNet-50'} 
model_name = 'ResNet-50' 
ckpt_path_list = ['ckpt/resnet50_best.pth', 'ckpt/resnet50_best.pth']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 



def main(): 
    if dataset_name == 'CIFAR-10':
        train_loader, test_loader = cifa10_data_load() 
        OUTPUT_DIM = 10 
    elif dataset_name == 'CIFAR-100':
        train_loader, test_loader = cifa100_data_load() 
        OUTPUT_DIM = 100 
    elif dataset_name == 'ImageNet': 
        train_loader, test_loader = imagenet_data_load() 
        OUTPUT_DIM = 1000
    else: 
        print('dataset name setting error!')
        return 
    
    if model_name == 'ResNet-50':
        model = resnet50() 
        IN_FEATURES = model.fc.in_features 
        final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM) 
        model.fc = final_fc  
    elif model_name == 'VggNet-11':
        model = vgg11_bn() 
        IN_FEATURES = model.classifier[-1].in_features
        final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM) 
        model.classifier[-1] = final_fc 
    else: 
        print('model name setting error!') 
    

    acc_list = []

    for idx in range(len(ckpt_path_list)): 
        state_dict = torch.load(ckpt_path_list[idx], map_location=None) 
        model.load_state_dict(state_dict)
        model = model.to(device) 

        model.eval() 
        acc = .0 
        time_stamp = 0 

        with tqdm(desc='Model %d - evaluation' % idx, unit='it', total=len(test_loader)) as pbar: 
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
        acc_list.append(val_acc) 
    
    print('[ACC] Mean: ', np.mean(acc_list), ' Std: ', np.std(acc_list))


if __name__ == '__main__': 
    main()