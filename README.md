# Meta-Ensemble Parameter Learning

This is the PyTorch implementation for inference and training of the weightformer network as described in: 
> **Meta-Ensemble Parameter Learning** 

In between, we introduce a new task, referred to **meta ensemble parameter learning**, which aims to directly predict the parameters of ensemble distillation model based on the parameters of base learners as well as small part of training dataset.  



## 🔥  WeightFormer

We introduce WeightFormer, a model to directly predict the distilled student model parameters. Our architecture takes inspiration from the Transformer and incorporates three key novelties to imitate the characteristics of model ensemble, i.e., cross-layer information flow, learnable attention mask and shift consistency limitation. 

<p align="center">
     <img src="https://github.com/feizc/Meta-Ensemble/blob/main/images/frame_weightformer.jpg" alt="Weightformer Architecture">
     <br/>
     <sub><em>
      Overview of WeightFormer for the generation of one layer weights. <br/> 
      Transformer-based weight generator receives concatenated weight matrices of teacher models along with model id and position information and produce the corresponding layer weights. After being generated, the predicted student model is used to compute the loss on the training set, whose gradients are then used to update the weights of WeightFormer. 
    </em></sub>
</p>



## ⚙  Dataset and Model

We support the image classification datasets: [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz), [CIFAR-100](https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz), and [ImageNet](http://image-net.org), for performance evaluation. Please download the corresponding datasets and put them in the file path data/ or specify with argparse.  

The trained checkpoints for WeighFormer will be available at Googledrive. 




## 🙌 Training 

Training scripts for different training scenarios. 

All the training scripts are in the folder `./scripts` and run `python script_name.py` for corresponding process. 

| Scripts      | Scenarios |
|--------------|-----------|
| train_vgg.py |  train single vggnet-11  | 
| train_resnet.py | train single resnet-50 | 
| train_distillation.py | average knowledge distillation for model ensemble |
| train_mlp.py | mlp network for weight generation | 
| train_transformer.py | WeightFormer for weight generation | 



For help or issues related to this package, please submit a GitHub issue. 


