#!/bin/bash
#
# The script to run multilayer perceptron with one hidden layer with specific
# set of parameters

~/anaconda3/bin/Rscript src/nn_analysis.R --learning_rate 0.00001 \
                                          --max_steps 50000 \
                                          --layers 512 \
                                          --batch_size 100 \
                                          --dropout 0.5 \
                                          --lr_anneal_step 10000 \
                                          --network_type mlp \
                                          --data_features_file $1
