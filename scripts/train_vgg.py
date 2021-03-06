# refer from https://colab.research.google.com/github/bentrevett/pytorch-image-classification/blob/master/4_vgg.ipynb#scrollTo=bnCcgZHXVwJq
from cgi import test
from torch import nn 
from torchvision import datasets, transforms 
import torch 
import numpy as np 
import matplotlib.pyplot as plt 
import random 
from tqdm import tqdm 
import os 

from model import vgg11_bn


batch_size = 64
epochs = 200
pretrain_path = 'ckpt/vgg11_bn.pth'
best_path = 'ckpt/best.pth' 
output_path = './ckpt'
OUTPUT_DIM = 100
load_pretrain = True 
load_ckpt = True


SEED = 2021

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def train():
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225]

    train_transform = transforms.Compose([
                               transforms.Resize(pretrained_size),
                               transforms.RandomRotation(5),
                               transforms.RandomHorizontalFlip(0.5),
                               transforms.RandomCrop(pretrained_size, padding=10),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=pretrained_means,
                                                    std=pretrained_stds)
                        ])

    test_transform = transforms.Compose([
                               transforms.Resize(pretrained_size),
                               transforms.ToTensor(),
                               transforms.Normalize(mean=pretrained_means,
                                                    std=pretrained_stds)
                        ])

    # https://pytorch.org/vision/stable/_modules/torchvision/datasets/cifar.html#CIFAR10 
    train_set = datasets.CIFAR10('data/cifar', train=True, download=False, transform=train_transform)
    print(len(train_set))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True) 
    
    
    test_set = datasets.CIFAR10('data/cifar', train=False, download=False, transform=test_transform)
    print(len(test_set))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False) # num_workers=2 

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # print(train_set.class_to_idx)
    # print the image example 
    # plt.imshow(np.transpose(train_set[0][0].numpy(), (1,2,0)))
    # plt.show()

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") 

    model = vgg11_bn() 
    if load_pretrain == True:
        state_dict = torch.load(pretrain_path, map_location=None) 
        model.load_state_dict(state_dict) 
    IN_FEATURES = model.classifier[-1].in_features
    final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM) 
    model.classifier[-1] = final_fc 
    
    model = model.to(device)
    # freeze parameter
    #for parameter in model.classifier[:-1].parameters():
    #    parameter.requires_grad = False

    loss_fn = nn.CrossEntropyLoss() 
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01,  momentum=0.9) 
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    
    #optimizer = torch.optim.SGD(model.parameters(), lr=0.01,
    #                  momentum=0.9, weight_decay=5e-4)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
    if load_ckpt == True: 
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
        
        # training 
        model.train()
        running_loss = 0.0 
        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(train_loader)) as pbar:
            for it, (image, label) in enumerate(train_loader):
                image, label = image.to(device), label.to(device) 
                # print(label.size())
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
                #if it == 20:
                #    break 
        # scheduler.step()
        val_acc = acc / time_stamp
        print(val_acc) 

        if val_acc > best_acc: 
            best_acc = val_acc 
            patience = 0
            torch.save(model.state_dict(), best_path) 
        else:
            patience += 1 


if __name__ == '__main__': 
    train()

