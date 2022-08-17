import argparse
import os 
import wget
import torch 
import math 
import clip 

from utils import cifa100_data_load

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

device = "cuda" if torch.cuda.is_available() else "cpu"  

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-location",
        type=str,
        default=os.path.expanduser('./data'),
        help="The root directory for the datasets.",
    )
    parser.add_argument(
        "--model-location",
        type=str,
        default=os.path.expanduser('./ckpt/vit'),
        help="Where to download the models.",
    )
    parser.add_argument(
        "--download-models", action="store_true", default=True,
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
    )
    return parser.parse_args()


# Utilities to make nn.Module functional
def del_attr(obj, names):
    if len(names) == 1:
        delattr(obj, names[0])
    else:
        del_attr(getattr(obj, names[0]), names[1:]) 


def set_attr(obj, names, val):
    if len(names) == 1:
        setattr(obj, names[0], val)
    else:
        set_attr(getattr(obj, names[0]), names[1:], val)



def make_functional(mod):
    orig_params = tuple(mod.parameters())
    # Remove all the parameters in the model
    names = []
    for name, p in list(mod.named_parameters()):
        del_attr(mod, name.split("."))
        names.append(name) 
    return orig_params, names


def load_weights(mod, names, params):
    for name, p in zip(names, params):
        set_attr(mod, name.split("."), p)


class AlphaWrapper(torch.nn.Module):
    def __init__(self, paramslist, model, names):
        super(AlphaWrapper, self).__init__()
        self.paramslist = paramslist
        self.model = model
        self.names = names
        ralpha = torch.ones(len(paramslist[0]), len(paramslist))
        ralpha = torch.nn.functional.softmax(ralpha, dim=1)
        self.alpha_raw = torch.nn.Parameter(ralpha)
        self.beta = torch.nn.Parameter(torch.tensor(1.))

    def alpha(self):
        return torch.nn.functional.softmax(self.alpha_raw, dim=1)

    def forward(self, inp):
        alph = self.alpha()
        params = tuple(sum(tuple(pi * alphai for pi, alphai in zip(p, alph[j].cpu()))) for j, p in enumerate(zip(*self.paramslist)))
        params = tuple(p.to(device) for p in params)
        load_weights(self.model, self.names, params)
        out = self.model(inp)
        return self.beta * out



class ModelWrapper(torch.nn.Module):
    def __init__(self, model, feature_dim, num_classes, normalize=False, initial_weights=None):
        super(ModelWrapper, self).__init__()
        self.model = model
        self.classification_head = torch.nn.Linear(feature_dim, num_classes)
        self.normalize = normalize
        if initial_weights is None:
            initial_weights = torch.zeros_like(self.classification_head.weight)
            torch.nn.init.kaiming_uniform_(initial_weights, a=math.sqrt(5))
        self.classification_head.weight = torch.nn.Parameter(initial_weights.clone())
        self.classification_head.bias = torch.nn.Parameter(
            torch.zeros_like(self.classification_head.bias))

        # Note: modified. Get rid of the language part.
        if hasattr(self.model, 'transformer'):
            delattr(self.model, 'transformer')

    def forward(self, images, return_features=False):
        features = self.model.encode_image(images)
        if self.normalize:
            features = features / features.norm(dim=-1, keepdim=True)
        logits = self.classification_head(features)
        if return_features:
            return logits, features
        return logits


def main(): 
    args = parse_arguments()
    NUM_MODELS = 72

    # Step 1: Download models.
    if args.download_models:
        if not os.path.exists(args.model_location):
            os.mkdir(args.model_location)
        for i in range(NUM_MODELS):
            print(f'\nDownloading model {i} of {NUM_MODELS - 1}')
            wget.download(
                f'https://github.com/mlfoundations/model-soups/releases/download/v0.0.2/model_{i}.pt',
                out=args.model_location
                )
    model_paths = [os.path.join(args.model_location, f'model_{i}.pt') for i in range(NUM_MODELS)] 
    base_model, preprocess = clip.load('./ckpt/clip/ViT-B-32.pt', device='cpu') 
    criterion = torch.nn.CrossEntropyLoss() 

    train_loader, test_loader = cifa100_data_load()
    
    sds = [torch.load(cp, map_location='cpu') for cp in model_paths] 
    feature_dim = sds[0]['classification_head.weight'].shape[1] # 512 
    num_classes = sds[0]['classification_head.weight'].shape[0]  # 1000
    model = ModelWrapper(base_model, feature_dim, num_classes, normalize=True)  # adjust classification head 
    model = model.to(device) 
    
    _, names = make_functional(model) 
    first = False 
    paramslist = [tuple(v.detach().requires_grad_().cpu() for _, v in sd.items()) for i, sd in enumerate(sds)]
    alpha_model = AlphaWrapper(paramslist, model, names) 

    lr = 0.05
    epochs = 1

    optimizer = torch.optim.AdamW(alpha_model.parameters(), lr=lr, weight_decay=0.) 

    for epoch in range(epochs): 
        for i, batch in enumerate(train_loader): 
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device) 
            optimizer.zero_grad() 
            out = alpha_model(inputs) 
            loss = criterion(out, labels) 
            loss.backward() 
            optimizer.step() 
            print(loss)
            break
        


if __name__ == '__main__': 
    main() 