import pickle
from random import shuffle 
from PIL import Image
from model import vgg11_bn 
from torch import nn 
import torch 
from torchvision import datasets, transforms 

from net_parameter import vgg11_predict_layers, resnet50_predict_layers 


def image_preprocess_transform():
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
    return train_transform, test_transform 



def cifa10_data_load(data_path='data/cifar', batch_size=8):
    # image transform 
    train_transform, test_transform = image_preprocess_transform() 

    train_set = datasets.CIFAR10(data_path, train=True, download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True) 
    
    test_set = datasets.CIFAR10(data_path, train=False, download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False) # num_workers=2 

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

    return train_loader, test_loader




def imagenet_data_load(data_path='data/imagenet', batch_size=8): 
    train_transform, test_transform = image_preprocess_transform() 

    train_set = datasets.ImageNet(data_path, split='train', transform=train_transform, download=False) 
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True) 

    val_set = datasets.ImageNet(data_path, split='val', transform=test_transform, download=False) 
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False) 

    return train_loader, val_loader 




def data_plot():
    cifar10_path = 'data/cifar-10/data_batch_1' 
    with open(cifar10_path, 'rb') as f: 
        dict = pickle.load(f, encoding='bytes') 

    img = dict[b'data'][1].reshape(3, 32, 32).transpose(1,2,0)
    im = Image.fromarray(img) 
    im.show() 


def ckpt_load(ckpt_path): 
    OUTPUT_DIM = 10

    state_dict = torch.load(ckpt_path, map_location=None) 
    model = vgg11_bn() 
    model.load_state_dict(state_dict) 
    IN_FEATURES = model.classifier[-1].in_features
    final_fc = nn.Linear(IN_FEATURES, OUTPUT_DIM) 
    model.classifier[-1] = final_fc 

    #print(model) 
    #print(model.state_dict().keys())
    #for key in model.state_dict().keys():
    #    print(key, model.state_dict()[key].size()) 
    return model 


def weight_dict_print(weight_dict): 
    for key in weight_dict.keys(): 
        print(key, weight_dict[key].size())


def weight_size_dict_generate(weight_dict): 
    weight_size_dict = {} 
    for key in weight_dict.keys(): 
        weight_size_dict[key] = weight_dict[key].size() 
    return weight_size_dict 


# combine the model weights in one dict  network={'vgg11', 'resnet50'}
def parameter_dict_combine(weight_dict_list, device, mode='linear', network='vgg11'): 
    combine_weight_dict = {} 
    weight_size_dict = {}

    if len(weight_dict_list) < 1: 
        return combine_weight_dict, weight_size_dict 

    for key in weight_dict_list[0].keys(): 
        if network == 'vgg11' and key not in vgg11_predict_layers:
            continue 
        if network == 'resnet50' and key not in resnet50_predict_layers:
            continue 
        weight_matrix = weight_dict_list[0][key] 
        if len(weight_matrix.size()) < 1:
            weight_matrix = weight_matrix.unsqueeze(0)
        out_dim = weight_matrix.size()[0] 
        weight_size_dict[key] = weight_matrix.size()
        weight_matrix = weight_matrix.view(out_dim, -1)
        
        for i in range(1, len(weight_dict_list)): 
            t_weight_matrix = weight_dict_list[i][key]
            t_weight_matrix = t_weight_matrix.view(out_dim, -1) 
            if mode == 'transformer': 
                weight_matrix = torch.cat((weight_matrix, t_weight_matrix), dim=0).to(device)
            else:
                weight_matrix = torch.cat((weight_matrix, t_weight_matrix), dim=-1).to(device)
        
        combine_weight_dict[key] = weight_matrix 
    return combine_weight_dict, weight_size_dict 
 

 # resize the weight dict for model loading 
def weight_resize_for_model_load(weight_dict, original_weight_dict, device): 
    for key in original_weight_dict.keys(): 
        if key in weight_dict.keys():
            weight_dict[key] = weight_dict[key].view(original_weight_dict[key].size()).to(device)
        else:
            weight_dict[key] = original_weight_dict[key].to(device)
    return weight_dict

# weight detach 
def weight_detach(weight_dict): 
    new_weight_dict = {} 
    for key in weight_dict.keys(): 
        new_weight_dict[key] = weight_dict[key].clone().detach() 
    return new_weight_dict



if __name__ == '__main__': 
    ckpt_path = 'ckpt/vgg11_bn.pth' 
    ckpt_load(ckpt_path)