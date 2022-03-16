# python -m torch.distributed.launch --nproc_per_node=2 train_resnet_ddp.py
import os 
import torch 
from torch import nn 
import torch.backends.cudnn as cudnn 
from tqdm import tqdm 
import random 
import numpy as np 

import argparse 
import torch.distributed as dist 
from torch.nn.parallel import DistributedDataParallel as DDP


from model import resnet50 
from utils import cifa10_data_load, weight_dict_print

batch_size = 8 
epochs = 200 
pretrain_path = 'ckpt/resnet50.pth' 
save_path = 'ckpt/resnet50_best.pth' 
OUTPUT_DIM = 10 
load_pretrain = True 
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 
cudnn.benchmark = True 

SEED = 2021

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True



def train(train_loader, model, optimizer, loss_fn, epoch, device): 
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
            break 
        

def validation(test_loader, model, epoch, device): 
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

    parser = argparse.ArgumentParser() 
    parser.add_argument("--local_rank", type=int, default=-1) 
    args = parser.parse_args() 
    local_rank = args.local_rank 
    
    if args.local_rank != -1:
        dist.init_process_group(backend='nccl', init_method='env://')
        torch.cuda.set_device(args.local_rank)
    map_location = "cuda:" + str(args.local_rank)
    device = torch.device('cuda', local_rank) 


    train_loader, test_loader = cifa10_data_load(batch_size=batch_size, distribution=True) 
    
    model = resnet50() 
    weight_dict_print(model.state_dict()) 

    if load_pretrain == True: 
        state_dict = torch.load(pretrain_path, map_location=map_location) 
        model.load_state_dict(state_dict) 
    IN_FEATURES = model.fc.in_features 
    final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM) 
    model.fc = final_fc 

    model = model.to(device) 
    # load weights before ddp 
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True) 

    
    loss_fn = nn.CrossEntropyLoss()  
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) 

    best_acc = 0.0 
    patience = 0 

    for epoch in range(epochs): 
        train(train_loader, model, optimizer, loss_fn, epoch, device) 
        val_acc = validation(test_loader, model, epoch, device) 
        print(val_acc) 

        if val_acc > best_acc: 
            best_acc = val_acc 
            patience = 0
            if args.local_rank == 0:
                torch.save(model.module.state_dict(), save_path) 
        else:
            patience += 1 




if __name__ == '__main__':
    main()