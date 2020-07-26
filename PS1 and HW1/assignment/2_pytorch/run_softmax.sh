#!/bin/sh
#############################################################################
# TODO: Modify the hyperparameters such as hidden layer dimensionality, 
#       number of epochs, weigh decay factor, momentum, batch size, learning 
#       rate mentioned here to achieve good performance
#############################################################################
python -u train.py \
    --model softmax \
    --epochs 50 \
    --weight-decay 0.96 \
    --momentum 0.92 \
    --batch-size 256 \
    --lr 1e-4 | tee softmax.log
#############################################################################
#                             END OF YOUR CODE                              #
#############################################################################
