import torch 
import torch.nn as nn
from torch.nn.modules.module import Module
from vgg import vgg11_bn 
import copy 
from torchvision import datasets, transforms  
from tqdm import tqdm 



class ModelEnsemble(nn.Module): 
    def __init__(self, model, weight_files, device):
        super(ModelEnsemble, self).__init__() 
        self.n = len(weight_files) 
        self.models = nn.ModuleList([copy.deepcopy(model) for _ in range(self.n)]) 
        for i in range(self.n): 
            state_dict_i = torch.load(weight_files[i]) 
            self.models[i].load_state_dict(state_dict_i)
            self.models[i] = self.models[i].to(device)

    def forward(self, image): 
        out_ensemble = [] 
        for i in range(self.n): 
            out_i = self.models[i](image) 
            out_ensemble.append(out_i.unsqueeze(0)) 
        return torch.mean(torch.cat(out_ensemble, 0), dim=0)



if __name__ == '__main__':  
    
    batch_size = 16
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu") 

    weight_files = ['ckpt/vgg11_bn.pth', 'ckpt/vgg11_bn.pth'] 
    ensemble_model = ModelEnsemble(vgg11_bn(), weight_files, device)  
    
    pretrained_size = 224
    pretrained_means = [0.485, 0.456, 0.406]
    pretrained_stds = [0.229, 0.224, 0.225] 

    test_transform = transforms.Compose([
                           transforms.Resize(pretrained_size),
                           transforms.ToTensor(),
                           transforms.Normalize(mean=pretrained_means,
                                                std=pretrained_stds)
                    ]) 
    
    test_set = datasets.CIFAR10('data', train=False, download=False, transform=test_transform) 
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False) # num_workers=2 

    ensemble_model.eval() 
    acc = .0 
    time_stamp = 0 
    with tqdm(desc='Ensemble evaluation', unit='it', total=len(test_loader)) as pbar:
        for it, (image, label) in enumerate(test_loader): 
            image, label = image.to(device), label.to(device) 
            # print(image.size(), label.size()) 
            with torch.no_grad(): 
                out = ensemble_model(image) 
                # print(out.size()) # (bsz, class) 
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



