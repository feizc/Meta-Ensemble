# Meta-Ensemble Parameter Learning

This is the PyTorch implementation for inference and training of the weightformer network as described in: 
> **Meta-Ensemble Parameter Learning** 

In between, we introduce a new task, referred to **meta ensemble parameter learning**, which aims to directly predict the parameters of ensemble distillation model based on the parameters of base learners as well as small part of training dataset.  

## 🔥  WeightFormer

we introduce WeightFormer, a model to directly predict the distilled student model parameters. Our architecture takes inspiration from the Transformer and incorporates three key novelties to imitate the characteristics of model ensemble, i.e., cross-layer information flow, learnable attention mask and shift consistency limitation. 

<p align="center">
     <img src="https://github.com/feizc/Meta-Ensemble/blob/main/images/frame_weightformer.jpg" alt="Weightformer Architecture">
     <br/>
     <sub><em>
      Overview of WeightFormer for the generation of one layer weights. <br/> 
      Transformer-based weight generator receives concatenated weight matrices of teacher models along with model id and position information and produce the corresponding layer weights. After being generated, the predicted student model is used to compute the loss on the training set, whose gradients are then used to update the weights of WeightFormer. 
    </em></sub>
</p>




## ⚙  Dataset 

We support the image classification datasets: CIFAR-10, CIFAR-100, and ImageNet, for performance evaluation. Please the datasets in the file path data/ or specify with argparse.  



## 🙌 Training 

Training scripts for different scenarios. 

