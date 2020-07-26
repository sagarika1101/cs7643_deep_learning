Name: Sagarika Srishti
email: ssrishti3@gatech.edu
Best Accuracy: 0.958

For this task, I used the pretrained weights of inception_v3 model, that I downloaded from the github repo https://github.com/huyvnphan/PyTorch-CIFAR10 .

I made a model using these weights, and then trained them from 5 epochs. Because of the weights being pretrained, the model started showing good accuracy even after 1 epoch. 

The hyperparameters that gave me the best accuracy were:
kernel size: 7
hidden dim: 64
weight_decay: 0.0
momentum: 0.0
batch_size: 128
learning_rate: 0.005

