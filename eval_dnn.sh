#!/bin/bash
#
# The evaluation script for experiment with Deep Learning Network
#

~/anaconda3/bin/Rscript src/nn_analysis.R --learning_rate 0.0001 \
                                          --max_steps 50000 \
                                          --layers 1024,512 \
                                          --batch_size 100 \
                                          --dropout 0.5 \
                                          --lr_anneal_step 10000 \
                                          --network_type dnn \
                                          --data_features_file $1