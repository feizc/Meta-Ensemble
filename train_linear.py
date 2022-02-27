import torch 
from torchvision import datasets, transforms 
from torch import nn 
from tqdm import tqdm 
from model import ParameterProject, vgg11_bn   
from utils import parameter_dict_combine, weight_dict_print, weight_size_dict_generate, weight_resize_for_model_load
from utils import weight_detach


batch_size = 8 
epochs = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
save_path = 'ckpt/parameter_predictor_best.pth' 

def main(): 

    # 1. download ckpt 
    ckpt_path_list = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 
    weight_dict_list = [] 
    for ckpt_path in ckpt_path_list:
        weight_dict_list.append(torch.load(ckpt_path, map_location=None)) 
    combine_weight_dict, weight_size_dict = parameter_dict_combine(weight_dict_list, device)
    weight_dict_print(weight_dict_list[0])
    combine_weight_size_dict = weight_size_dict_generate(combine_weight_dict)
    

    # 2. download dataset 
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

    # 3. initialize
    parameter_predict_model = ParameterProject(weight_size_dict=combine_weight_size_dict) 
    parameter_predict_model = parameter_predict_model.to(device) 
    vgg_model = vgg11_bn() 
    vgg_model = vgg_model.to(device)
    
    loss_fn = nn.CrossEntropyLoss() 
    optimizer = torch.optim.Adam(parameter_predict_model.parameters(), lr=0.0001) 

    best_acc = 0.0 
    patience = 0
    
    # 4. training
    for epoch in range(epochs): 
        parameter_predict_model.train()
        running_loss = 0.0 
        with tqdm(desc='Epoch %d - train' % epoch, unit='it', total=len(train_loader)) as pbar: 
            for it, (image, label) in enumerate(train_loader):
                image, label = image.to(device), label.to(device) 
                optimizer.zero_grad() 
                new_combine_weight_dict = weight_detach(combine_weight_dict)
                # combine_weight_dict, _ = parameter_dict_combine(weight_dict_list, device)
                # weight_dict_print(combine_weight_dict)
                generated_weight_dict = parameter_predict_model(new_combine_weight_dict)
                generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device) 
                # model parameter load
                vgg_model.load_state_dict(generated_weight_dict) 
                out = vgg_model(image) 
                loss = loss_fn(out, label) 
                loss.backward() 
                optimizer.step()

                running_loss += loss.item() 
                pbar.set_postfix(loss=running_loss / (it + 1))
                pbar.update() 
                break 
            
    
        # 5. validation 
        parameter_predict_model.eval() 
        acc = .0 
        time_stamp = 0
        with tqdm(desc='Epoch %d - evaluation' % epoch, unit='it', total=len(test_loader)) as pbar:
            new_combine_weight_dict = weight_detach(combine_weight_dict)
            generated_weight_dict = parameter_predict_model(new_combine_weight_dict)
            generated_weight_dict = weight_resize_for_model_load(generated_weight_dict, weight_dict_list[0], device) 
            # model parameter load
            vgg_model.load_state_dict(generated_weight_dict) 
            for it, (image, label) in enumerate(test_loader): 
                image, label = image.to(device), label.to(device) 
                with torch.no_grad(): 
                    out = vgg_model(image) # (bsz, vob)
                    predict_y = torch.max(out, dim=1)[1] #(bsz, ) 
                    acc += (predict_y == label).sum().item() / predict_y.size(0)
                pbar.set_postfix(acc=acc / (it + 1))
                pbar.update() 
                time_stamp += 1 
                break
        # scheduler.step()
        val_acc = acc / time_stamp
        print(val_acc) 

        if val_acc > best_acc: 
            best_acc = val_acc 
            patience = 0
            torch.save(parameter_predict_model.state_dict(), save_path) 
        else:
           patience += 1 


if __name__ == '__main__': 
    main() 
