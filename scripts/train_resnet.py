import os 
import torch 
from torch import nn 
import torch.backends.cudnn as cudnn 
from tqdm import tqdm 
import random 
import numpy as np 

from model import resnet50 
from utils import cifa10_data_load, weight_dict_print

batch_size = 8 
epochs = 200 
pretrain_path = 'ckpt/resnet50.pth' 
best_path = 'ckpt/resnet50_best.pth' 
output_path = './ckpt'
OUTPUT_DIM = 10 
load_pretrain = True 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
resume_last = True 

cudnn.benchmark = True 

SEED = 2021

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def train(train_loader, model, optimizer, loss_fn, epoch): 
    model.train() 
    running_loss = 0.0 
    with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(train_loader)) as pbar:
        for it, (image, label) in enumerate(train_loader):
            image, label = image.to(device), label.to(device) 
            #print(label.size())
            optimizer.zero_grad()
            out = model(image) 
            loss = loss_fn(out, label) 
            loss.backward() 
            optimizer.step()

            running_loss += loss.item() 
            pbar.set_postfix(loss=running_loss / (it + 1))
            pbar.update() 
            if it % 10000 == 0:
                torch.save({
                    'torch_rng_state': torch.get_rng_state(),
                    # 'cuda_rng_state': torch.cuda.get_rng_state(),
                    'numpy_rng_state': np.random.get_state(),
                    'random_rng_state': random.getstate(),
                    'epoch': epoch,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }, os.path.join(output_path, "latest.pth"),
                )
            break 
        


def validation(test_loader, model, epoch): 
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



def main(): 
    train_loader, test_loader = cifa10_data_load(batch_size=batch_size) 
    model = resnet50() 
    weight_dict_print(model.state_dict()) 

    if load_pretrain == True: 
        state_dict = torch.load(pretrain_path, map_location=None) 
        model.load_state_dict(state_dict) 
    IN_FEATURES = model.fc.in_features 
    final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM) 
    model.fc = final_fc 
    
    model = model.to(device) 
    
    loss_fn = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 


    if resume_last == True: 
        fname = os.path.join(output_path, "latest.pth") 
        if os.path.exists(fname):
            data = torch.load(fname)
            torch.set_rng_state(data['torch_rng_state'])
            # torch.cuda.set_rng_state(data['cuda_rng_state'])
            np.random.set_state(data['numpy_rng_state'])
            random.setstate(data['random_rng_state'])
            model.load_state_dict(data['state_dict'], strict=False)
            optimizer.load_state_dict(data['optimizer']) 
            print('load last trained ckpt!')

    best_acc = 0.0 
    patience = 0 
    for epoch in range(epochs): 
        train(train_loader, model, optimizer, loss_fn, epoch) 
        val_acc = validation(test_loader, model, epoch) 
        print(val_acc) 

        if val_acc > best_acc: 
            best_acc = val_acc 
            patience = 0
            torch.save(model.state_dict(), best_path) 
        else:
            patience += 1 




if __name__ == '__main__':
    main()
