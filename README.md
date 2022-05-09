# Meta Ensemble Parameter Learning
 
We introduce a new task, referred to **meta ensemble parameter learning**, which aims to directly predict the parameters of ensemble distillation model based on the parameters of base learners as well as small part of training dataset.  

## 🔥  WeightFormer

<p align="center">
     <img src="https://github.com/feizc/Meta-Ensemble/blob/main/images/frame_weightformer.jpg" alt="Weightformer Architecture">
     <br/>
     <sub><em>
      Overview of WeightFormer for the generation of one layer weights. <br/> 
      Transformer-based weight generator receives concatenated weight matrices of teacher models along with model id and position information and produce the corresponding layer weights. After being generated, the predicted student model is used to compute the loss on the training set, whose gradients are then used to update the weights of WeightFormer. 
    </em></sub>
</p>




## ⚙  Dataset 

We support the common image classification datasets: CIFAR-10, CIFAR-100, and ImageNet for performance evaluation. 


## 🙌 Training 

training script.
