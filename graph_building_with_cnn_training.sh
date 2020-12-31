#!/bin/sh
cd ./preprocessing

CONFIG_FILE='./configs/build_graph_with_feature_training.yaml'
WOKERS=8

python cnn_sample.py --cfg $CONFIG_FILE --num-workers $WOKERS

for((FOLD=0;FOLD<1;FOLD++)); 
do
    python cnn_train.py --cfg $CONFIG_FILE --epochs 30 --fold $FOLD\
        --batch-size 256 -j $WOKERS --weighted-sample\
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0

    python graph_construct.py --cfg $CONFIG_FILE --fold $FOLD\
        --batch-size 256 --num-workers $WOKERS\
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0

    python path_make_list.py --cfg $CONFIG_FILE --fold $FOLD
done