import pickle 
from PIL import Image 
from model import vgg11_bn 
import torch 


def data_plot():
    cifar10_path = 'data/cifar-10/data_batch_1' 
    with open(cifar10_path, 'rb') as f: 
        dict = pickle.load(f, encoding='bytes') 

    img = dict[b'data'][1].reshape(3, 32, 32).transpose(1,2,0)
    im = Image.fromarray(img) 
    im.show() 


def ckpt_load(ckpt_path): 
    state_dict = torch.load(ckpt_path, map_location=None) 
    print(state_dict)
    model = vgg11_bn() 
    model.load_state_dict(state_dict)



if __name__ == '__main__': 
    ckpt_path = 'ckpt/vgg11_bn.pth' 
    ckpt_load(ckpt_path)