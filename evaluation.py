# script for multiple checkpoints evaluation and report the statics results 
from torch import nn 
import torch  
from tqdm import tqdm 
import numpy as np 

from model import ensemble_model, vgg11_bn, resnet50, ModelEnsemble 
from train_vgg import OUTPUT_DIM


from utils import cifa10_data_load, cifa100_data_load, imagenet_data_load 

# {'CIFAR-10', 'CIFAR-100', 'ImageNet'}
dataset_name = 'CIFAR-10'
# {'VggNet-11', 'ResNet-50'} 
model_name = 'ResNet-50' 
ckpt_path_list = ['ckpt/resnet50_best.pth', 'ckpt/resnet50_best.pth']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
ensemble_flag = True 

# On Calibration of Modern Neural Networks 
def ece_score(py, y_test, n_bins=10):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


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
    
    if ensemble_flag == True: 
        ensemble_model = ModelEnsemble(model, ckpt_path_list, device) 


    acc_1_list = [] 
    acc_5_list = []
    ece_list = [] 
    f_softmax = nn.functional.softmax


    for idx in range(len(ckpt_path_list)): 
        state_dict = torch.load(ckpt_path_list[idx], map_location=None) 
        model.load_state_dict(state_dict)
        model = model.to(device) 

        model.eval() 
        acc = .0 
        acc5 = .0 
        ece = .0
        time_stamp = 0 

        with tqdm(desc='Model %d - evaluation' % idx, unit='it', total=len(test_loader)) as pbar: 
            for it, (image, label) in enumerate(test_loader): 
                image, label = image.to(device), label.to(device) 
                with torch.no_grad(): 
                    out = model(image) # (bsz, vob) 
                    predict_y = torch.max(out, dim=1)[1] #(bsz, ) 
                    predict_top_5 = torch.topk(out, dim=1, k=5)[1] 
                    ece += ece_score(f_softmax(out, dim=1).cpu().numpy(), label.cpu().numpy())
                    acc += (predict_y == label).sum().item() / predict_y.size(0) 
                    acc5 += (predict_top_5 == label.unsqueeze(1)).sum().item() / predict_top_5.size(0) 
                    
                pbar.set_postfix(acc=acc / (it + 1))
                pbar.update() 
                time_stamp += 1 
                break 
        val_acc1 = acc / time_stamp 
        val_acc5 = acc5 / time_stamp 
        val_ece = ece / time_stamp 

        acc_1_list.append(val_acc1) 
        acc_5_list.append(val_acc5)
        ece_list.append(val_ece)
    
    print('[ACC-1] Mean: ', np.mean(acc_1_list), ' Std: ', np.std(acc_1_list))
    print('[ACC-5] Mean: ', np.mean(acc_5_list), ' Std: ', np.std(acc_5_list))
    print('[ECE] Mean: ', np.mean(ece_list), ' Std: ', np.std(ece_list))

    if ensemble_flag == True:
        acc = .0 
        acc5 = .0 
        ece = .0
        time_stamp = 0 

        with tqdm(desc='Ensemble - evaluation', unit='it', total=len(test_loader)) as pbar: 
            for it, (image, label) in enumerate(test_loader): 
                image, label = image.to(device), label.to(device) 
                with torch.no_grad(): 
                    out = ensemble_model(image) # (bsz, vob) 
                    predict_y = torch.max(out, dim=1)[1] #(bsz, ) 
                    predict_top_5 = torch.topk(out, dim=1, k=5)[1] 
                    ece += ece_score(f_softmax(out, dim=1).cpu().numpy(), label.cpu().numpy())
                    acc += (predict_y == label).sum().item() / predict_y.size(0) 
                    acc5 += (predict_top_5 == label.unsqueeze(1)).sum().item() / predict_top_5.size(0) 
                    
                pbar.set_postfix(acc=acc / (it + 1))
                pbar.update() 
                time_stamp += 1 
                break 
        val_acc1 = acc / time_stamp 
        val_acc5 = acc5 / time_stamp 
        val_ece = ece / time_stamp 
        print('Ensemble Results:', val_acc1, val_acc5, val_ece)





if __name__ == '__main__': 
    main()