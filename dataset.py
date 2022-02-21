import torch
from torchvision import datasets, transforms 
from torch.utils.data import Dataset 
import json 
from torch import nn 
from tqdm import tqdm 
import numpy as np 

from model_ensemble import VggEnsemble 
from utils import cifa10_data_load
from model import vgg11_bn 


resize_flag = False 
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu") 
weight_files = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 


# ensemble prediction dataset 
class EnsembleDataset(Dataset): 
    def __init__(self, data_path):
        with open(data_path, 'r', encoding='utf-8') as f: 
            data_dict = json.load(f)
        self.data = torch.from_numpy(np.array(data_dict['image']))
        self.target = torch.from_numpy(np.array(data_dict['label']))
    
    def __len__(self):
        return len(self.data) 
    
    def __getitem__(self, index):
        return self.data[index], self.target[index] 



def vgg_ensemble_generate(): 
    train_loader, test_loader = cifa10_data_load() 
    base_model = vgg11_bn() 
    if resize_flag == True: 
        IN_FEATURES = base_model.classifier[-1].in_features
        final_fc = nn.Linear(IN_FEATURES, 10) 
        base_model.classifier[-1] = final_fc 
    ensemble_model = VggEnsemble(base_model, weight_files)  
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

    with open('data/ensemble_result.json', 'w', encoding='utf-8') as f: 
        json.dump(data_dict, f) 




if __name__ == '__main__': 
    # vgg_ensemble_generate() 
    data_path = 'data/ensemble_result.json' 
    dataset = EnsembleDataset(data_path) 
    # image, label = dataset[0] 
    #print(image.size(), label) 

    train_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True) 

    with tqdm(desc='Ensemble evaluation', unit='it', total=len(train_loader)) as pbar:
        for it, (image, label) in enumerate(train_loader): 
            print(image.size(), label.size()) 
            break 
        


