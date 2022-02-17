import torch 

from torch.optim.lr_scheduler import MultiStepLR 
from utils import cifa10_data_load
from model import EPTransformer, vgg11_bn 
import torch.nn as nn


class Trainer(): 
    def __init__(self, optimizer, n_batches, device, grad_clip=5):
        self.optimizer = optimizer 
        self.criterion = nn.CrossEntropyLoss().to(device)
        self.n_batches = n_batches 
        self.grad_clip = grad_clip 
        self.device = device
        self.reset() 
    
    def reset(self): 
        self.metrics = {} 
    

def main(): 
    # hyperparameters
    batch_size = 8 
    lr = 1e-3
    wd = 1e-5
    epochs = 300 
    lr_steps = '200,250'
    gamma = 0.1 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    
    train_loader, test_loader = cifa10_data_load(batch_size) 
    EPmodel = EPTransformer(N=3, padding_idx=0) 
    base_model = vgg11_bn() 
    
    optimizer = torch.optim.Adam(EPmodel.parameters(), lr=lr, weight_decay=wd) 
    scheduler = MultiStepLR(optimizer, milestones=lr_steps, gamma=gamma) 
    
    print('\nStarting training ensemble model parameter predictor!') 

    for epoch in range(epochs): 
        print('\nepoch={:03d}/{:03d}, lr={:e}'.format(epoch + 1, epochs, scheduler.get_last_lr()[0]))
        EPmodel.train() 
        for step, (images, targets) in enumerate(train_loader): 
            print(step)
            break

        scheduler.step() 
        break 


if __name__ == '__main__':
    main()
