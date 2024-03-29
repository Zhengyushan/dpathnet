## DRA-Net: Diagnostic Regions Attention Network for Histopathology WSI Recommendation and Retrieval

This is a PyTorch implementation of the paper [DRA-Net](https://doi.org/10.1109/TMI.2020.3046636):
```
@Article{zheng2020diagnostic,
  author  = {Zheng, Yushan and Jiang, Zhiguo and Shi, Jun and Xie, Fengying and Zhang, Haopeng and 
             Huai, Jianguo and Cao, Ming and Yang, Xiaomiao},
  title   = {Diagnostic Regions Attention Network (DRA-Net) for Histopathology WSI Recommendation and Retrieval},
  journal = {IEEE Transactions on Medical Imaging},
  year    = {2020},
  doi     = {10.1109/TMI.2020.3046636},
}
```
It also includes the implementation of the paper [DPathNet](https://doi.org/10.1007/978-3-030-59722-1_44):
```
@inproceedings{zheng2020tracing,
	author    = {Zheng, Yushan and Jiang, Zhiguo and Zhang, Haopeng and Xie, Fengying and Shi, Jun},
	title     = {Tracing Diagnosis Paths on Histopathology WSIs for Diagnostically Relevant Case Recommendation},
	booktitle = {Medical Image Computing and Computer-Assisted Intervention},
	year      = {2020},
        pages     = {459--469},
        doi       = {10.1007/978-3-030-59722-1_44},
}

```

### Preprocessing

We need to extract the features of image content under the diagnosis path first.

To extract the features using the CNN trained on the ImageNet dataset, please refer to
[graph_building.sh](./graph_building.sh)

To extract the features using the CNN trained by the pathologists' annotations, please refer to [graph_building_with_cnn_training.sh](./graph_building_with_cnn_training.sh):


### Training

To train the DRA-Net, run:
```
DATADIR = [the directory of the data list generated in the preprocessing step.]

for((FOLD=0;FOLD<5;FOLD++)); 
do
    python main.py --dataset-dir $DATADIR/list_fold_$FOLD --prefix-name DRA-Net\
        --hash-bits 32 --rnn-model GRU --num-rnn-layers 1\
        --num-epochs 300 --batch-size 32 --num-workers 8\
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
done
```

To train the DPathNet, run:
```
DATADIR = [the directory of the data list generated in the preprocessing step.]

for((FOLD=0;FOLD<5;FOLD++)); 
do
    python main.py --dataset-dir $DATADIR/list_fold_$FOLD --prefix-name DPathNet\
        --hash-bits 32 --rnn-model GRU --num-rnn-layers 1\
        --disable-att --num-epochs 300 --batch-size 32 --num-workers 8\
        --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0
done
```

### Data description
The structure of the whole slide image dataset to run the code.
```
./data                                    # The directory of the data.
├─ 0A00DD22-A08E-4B47-A51B-94A8BD039DAA   # The directory for a slide, which is named by GUID in our dataset.
│  ├─ Large                               # The directory of image tiles in Level 0 (40X lens).
│  │  ├─ 0000_0000.jpg                    # The image tile in Row 0 and Column 0.
│  │  ├─ 0000_0001.jpg                    # The image tile in Row 0 and Column 1.
│  │  └─ ...
│  ├─ Medium                              # The directory of image tiles in Level 1 (20X lens).
│  │  ├─ 0000_0000.jpg
│  │  ├─ 0000_0001.jpg
│  │  └─ ...
│  ├─ Small                               # The directory of image tiles in Level 2 (10X lens).
│  │  ├─ 0000_0000.jpg
│  │  ├─ 0000_0001.jpg
│  │  └─ ...
│  ├─ Overview                            # The directory of image tiles in Level 3 (5X lens).
│  │  ├─ 0000_0000.jpg
│  │  ├─ 0000_0001.jpg
│  │  └─ ...
│  ├─ Overview.jpg                        # The thumbnail of the WSI in Level 3.          
│  ├─ AnnotationMask.png                  # The pixel-wise annotation mask of the WSI in Level 3.
│  └─ BrowsingRecord.pkl                  # The file to store the sequence of browing screens by 
│                                           coordinates ((left, right, top, bottom),...) in Level 3.
├─ 0A003711-3BE4-44E2-9280-89D84E5AF59F
└─ ...
```