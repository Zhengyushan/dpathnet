#!/bin/sh
cd ./preprocessing

CONFIG_FILE='./configs/build_graph_imagenet_feature.yaml'

python graph_construct.py --cfg $CONFIG_FILE --num-workers 8 --batch-size 256\
    --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0

python path_make_list.py --cfg $CONFIG_FILE
