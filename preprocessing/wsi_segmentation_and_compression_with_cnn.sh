# This processing flow is used to segment & compress WSI with a trained CNN.
# It is not a part of the DRA-Net or DPathNet.
# We reserve the flow with the script "cnn_wsi_encode.py" here for possible
# usage in the future studies.

CONFIG_FILE='./configs/build_graph_with_feature_training.yaml'
WOKERS=20

python cnn_sample.py --cfg $CONFIG_FILE --num-workers $WOKERS

for((FOLD=0;FOLD<5;FOLD++)); 
do
    python cnn_train.py --cfg $CONFIG_FILE --epochs 100 --fold $FOLD\
        --batch-size 256 -j $WOKERS --weighted-sample\
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0

    python cnn_wsi_encode.py --cfg $CONFIG_FILE --fold $FOLD\
        --batch-size 1024 --num-workers $WOKERS\
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
done