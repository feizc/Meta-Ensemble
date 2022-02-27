import torch
from torchvision import datasets, transforms 
from torch.utils.data import Dataset 
import json 
from torch import nn 
from tqdm import tqdm 
import numpy as np 

from model import ModelEnsemble, vgg11_bn
from utils import cifa10_data_load




# ensemble prediction dataset 
class EnsembleDataset(Dataset): 
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f: 
            data_dict = json.load(f)
        self.data = torch.from_numpy(np.array(data_dict['image'])).float()
        self.target = torch.from_numpy(np.array(data_dict['label'])).long()
    
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, index):
        return self.data[index], self.target[index] 



def vgg_ensemble_generate(ensemble_train_path, ensemble_test_path, resize_flag, weight_files, device): 
    train_loader, test_loader = cifa10_data_load() 
    base_model = vgg11_bn() 
    if resize_flag == True: 
        IN_FEATURES = base_model.classifier[-1].in_features
        final_fc = nn.Linear(IN_FEATURES, 10) 
        base_model.classifier[-1] = final_fc 
    ensemble_model = ModelEnsemble(base_model, weight_files, device)  
    image_m = None
    label_m = None

    data_dict = {}
    with tqdm(desc='Ensemble prediction for train', unit='it', total=len(train_loader)) as pbar:
        for it, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device) 
            #print(image.size())
            with torch.no_grad(): 
                out = ensemble_model(image) 
                predict_y = torch.max(out, dim=1)[1] #(bsz, ) 
                #print(predict_y.size(), predict_y)
            if it == 0: 
                image_m = image 
                label_m = predict_y 
            else: 
                image_m = torch.cat((image_m, image), 0) 
                label_m = torch.cat((label_m, predict_y),0)
            pbar.update() 
            if it == 16:
                break 
    print(image_m.size()) 
    print(label_m.size()) 
    data_dict['image'] = image_m.cpu().numpy().tolist()
    data_dict['label'] = label_m.cpu().numpy().tolist()

    with open(ensemble_train_path, 'w', encoding='utf-8') as f: 
        json.dump(data_dict, f) 


    with tqdm(desc='Ensemble prediction for test', unit='it', total=len(test_loader)) as pbar:
        for it, (image, label) in enumerate(test_loader):
            image, label = image.to(device), label.to(device) 
            #print(image.size())
            with torch.no_grad(): 
                out = ensemble_model(image) 
                predict_y = torch.max(out, dim=1)[1] #(bsz, ) 
                #print(predict_y.size(), predict_y)
            if it == 0: 
                image_m = image 
                label_m = predict_y 
            else: 
                image_m = torch.cat((image_m, image), 0) 
                label_m = torch.cat((label_m, predict_y),0)
            pbar.update() 
            if it == 16:
                break 
    print(image_m.size()) 
    print(label_m.size()) 
    data_dict['image'] = image_m.cpu().numpy().tolist()
    data_dict['label'] = label_m.cpu().numpy().tolist()

    with open(ensemble_test_path, 'w', encoding='utf-8') as f: 
        json.dump(data_dict, f) 




if __name__ == '__main__': 
    resize_flag = False 
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
    weight_files = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 
    train_data_path = 'data/ensemble_train_data.json'
    test_data_path = 'data/ensembe_test_data.json' 
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
    weight_files = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 
    
    # vgg_ensemble_generate(train_data_path, test_data_path, resize_flag, weight_files, device) 
    data_path = 'data/ensemble_train_data.json' 
    dataset = EnsembleDataset(data_path) 
    # image, label = dataset[0] 
    #print(image.size(), label) 

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True) 

    with tqdm(desc='Ensemble evaluation', unit='it', total=len(train_loader)) as pbar:
        for it, (image, label) in enumerate(train_loader): 
            print(image.size(), label.size()) 
            break 
        



