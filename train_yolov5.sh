#!/bin/bash

for fold in 4

do
    echo "FOLD NUMBER: $fold"
    python ./src/models/yolov5/train.py --img 512 \
                                        --batch 16 \
                                        --epochs 100 \
                                        --data src/models/yolov5/data/data_fold_${fold}.yaml \
                                        --weights ./models/pretrained_models/yolov5/yolov5x.pt \
                                        --save_period 1 \
                                        --project ./models/yolov5/batch16_ls02 \
                                        --label-smoothing 0.2 \
                                        --name yolov5x-img-512-fold-${fold}
    echo "###########################################################################################\n"
    sleep 3;
done

#                                        # --label-smoothing 0.2 \
