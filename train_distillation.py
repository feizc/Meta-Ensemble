# knowledge distillation for vgg 
from torch import nn  
import json 
import torch 
import os 
from tqdm import tqdm

from model import ModelEnsemble, vgg11_bn 
from utils import cifa10_data_load
from dataset import EnsembleDataset, vgg_ensemble_generate



train_data_path = 'data/ensemble_train_data.json'
test_data_path = 'data/ensembe_test_data.json' 
resize_flag = False 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
weight_files = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 
batch_size = 8
save_path = 'ckpt/vgg11_ditillation.pth'
epochs = 50



def main(): 
    # if generate the prediction results file 
    #if not os.path.exists(train_data_path):
    #    vgg_ensemble_generate(train_data_path, test_data_path, resize_flag, weight_files, device) 
    
    #ensemble_train_set = EnsembleDataset(train_data_path) 
    #ensemble_train_loader = torch.utils.data.DataLoader(ensemble_train_set, batch_size=batch_size, shuffle=True) 
    #test_set = EnsembleDataset(test_data_path) 
    #test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False) # num_workers=2  

    train_loader, test_loader = cifa10_data_load()

    model = vgg11_bn() 
    if resize_flag == True: 
        IN_FEATURES = model.classifier[-1].in_features
        final_fc = nn.Linear(IN_FEATURES, 10) 
        model.classifier[-1] = final_fc  
    ensemble_model = ModelEnsemble(model, weight_files, device)  
    model = model.to(device) 

    loss_fn = nn.CrossEntropyLoss() 
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

    best_acc = 0.0 
    patience = 0
    
    for epoch in range(epochs): 
        
        # training 
        model.train()
        running_loss = 0.0 
        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(train_loader)) as pbar:
            for it, (image, label) in enumerate(train_loader):
                image, label = image.to(device), label.to(device) 
                with torch.no_grad(): 
                    out = ensemble_model(image) 
                    predict_y = torch.max(out, dim=1)[1] 
                optimizer.zero_grad()
                out = model(image) 
                loss = loss_fn(out, predict_y) 
                loss.backward() 
                optimizer.step()

                running_loss += loss.item() 
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update() 
                if it == 20:
                    break 
        
        # validate 
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
                if it == 20:
                    break 
        # scheduler.step()
        val_acc = acc / time_stamp
        print(val_acc) 

        if val_acc > best_acc: 
            best_acc = val_acc 
            patience = 0
            torch.save(model.state_dict(), save_path) 
        else:
            patience += 1 




if __name__ == '__main__': 
    main()
