#!/bin/sh
DATADIR="./data/graph_list/[efficientnet-b0_pre][fs112_m500][pl64]"

for((FOLD=0;FOLD<5;FOLD++)); 
do
    python main.py --dataset-dir $DATADIR/list_fold_$FOLD --prefix-name DPathNet\
        --hash-bits 40 --rnn-model GRU --num-rnn-layers 1\
        --disable-att --num-epochs 300 --batch-size 32 --num-workers 4\
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
done