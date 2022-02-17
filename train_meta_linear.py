from appscript import con
import torch 
from torchvision import datasets, transforms 
from torch import nn 
from tqdm import tqdm 
from model import ParameterProject 
from utils import parameter_dict_combine, weight_dict_print, weight_detach,\
                    weight_size_dict_generate, weight_resize_for_model_load
from model import vgg11_bn 


device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
meta_epochs = 200
base_epochs = 8 
batch_size = 8
save_path = 'ckpt/linear_parameter_predictor_best.pth' 
# analyze how many ratio for training 
few_shot_flag = True 


def train(): 
    # 1. download ckpt 
    ckpt_path_list = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 
    weight_dict_list = [] 
    for ckpt_path in ckpt_path_list:
        weight_dict_list.append(torch.load(ckpt_path, map_location=None)) 
    combine_weight_dict, weight_size_dict = parameter_dict_combine(weight_dict_list, device) 
    combine_weight_size_dict = weight_size_dict_generate(combine_weight_dict) 

    # 2. meta learner and base learner 
    parameter_predict_model = ParameterProject(weight_size_dict=combine_weight_size_dict) 
    parameter_predict_model = parameter_predict_model.to(device) 
    base_model = vgg11_bn()
    base_model = base_model.to(device)

    meta_optimizer = torch.optim.Adam(parameter_predict_model.parameters(), lr=0.0001) 

    
    for epoch in range(meta_epochs): 
        parameter_predict_model.train() 
        new_combine_weight_dict = weight_detach(combine_weight_dict)
        generated_weight_dict = parameter_predict_model(new_combine_weight_dict) 
        generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device)
        detach_weight_dict = weight_detach(generated_weight_dict) 

        base_model.load_state_dict(detach_weight_dict) 
        
        target_weight_dict = train_base(base_model) 

        for t in range(base_epochs): 
            
            new_combine_weight_dict = weight_detach(combine_weight_dict)
            generated_weight_dict = parameter_predict_model(new_combine_weight_dict) 
            generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device)
            loss = meta_loss(generated_weight_dict, target_weight_dict) 
            meta_optimizer.zero_grad() 
            loss.backward() 
            meta_optimizer.step() 
            print('meta loss:', loss.item()) 
            # print(parameter_predict_model.state_dict())
        torch.save(parameter_predict_model.state_dict(), save_path) 
    


# jointly training tuning for performance boosting 
def combine_tuning(parameter_predict_model, base_model): 
    train_loader, test_loader = cifa10_data_load() 
    meta_optimizer = torch.optim.SGD(parameter_predict_model.parameters(), lr=0.1, momentum=0.9) 

    return base_model



def meta_loss(generated_weight_dict, target_weight_dict): 
    begin_flag = True 
    mse_loss = nn.MSELoss()
    for key in generated_weight_dict: 
        if 'features.0.weight' not in key: 
            continue 
        if begin_flag == True: 
            loss = mse_loss(generated_weight_dict[key].float(), target_weight_dict[key].float()) 
            begin_flag = False 
        else:
            loss += mse_loss(generated_weight_dict[key].float(), target_weight_dict[key].float())  
    return loss



# train for total/ only fc 
def train_base(base_model): 

    # dataset 
    train_loader, test_loader = cifa10_data_load()
    
    base_optimizer = torch.optim.Adam(base_model.parameters(), lr=0.0001) 
    loss_fn = nn.CrossEntropyLoss() 

    # train base 
    for epoch in range(base_epochs): 
        
        # validation the parameter prediction performance 

        base_model.train()
        running_loss = 0.0 
        total_part_num = len(train_loader) // 10 
        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(train_loader)) as pbar: 
            for it, (image, label) in enumerate(train_loader):
                image, label = image.to(device), label.to(device) 
                base_optimizer.zero_grad() 

                out = base_model(image) 
                loss = loss_fn(out, label) 
                loss.backward() 
                base_optimizer.step()

                running_loss += loss.item() 
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update() 
                if few_shot_flag == True: 
                    if it % total_part_num == 0: 
                        print('Train set partial: ', it // total_part_num) 
                        print('\n')
                        valid_base(base_model, test_loader, epoch)
                break 
        break
    
    return base_model.state_dict()



def valid_base(base_model, test_loader, epoch):
    acc = .0 
    time_stamp = 0 
    base_model.eval() 
    with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(test_loader)) as pbar:
        for it, (image, label) in enumerate(test_loader): 
            image, label = image.to(device), label.to(device) 
            with torch.no_grad(): 
                out = base_model(image) # (bsz, vob)
                predict_y = torch.max(out, dim=1)[1] #(bsz, ) 
                acc += (predict_y == label).sum().item() / predict_y.size(0)
            pbar.set_postfix(acc=acc / (it + 1))
            pbar.update() 
            time_stamp += 1 
            break 
        val_acc = acc / time_stamp
        print(val_acc) 
        print('\n')


def cifa10_data_load():
    # dataset 
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

    train_set = datasets.CIFAR10('data', train=True, download=False, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True) 
    
    test_set = datasets.CIFAR10('data', train=False, download=False, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False) # num_workers=2 

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck') 

    return train_loader, test_loader



if __name__ == '__main__': 
    train() 
