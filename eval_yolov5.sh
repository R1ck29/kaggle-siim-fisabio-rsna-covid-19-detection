#!/bin/bash

# CONFIDENCE = [
#     0.269, 0.268, 0.209, 0.179, 0.308
# ]

# CONFIDENCE = [
#     0.114, 0.143
# ]

for fold in 1

do
    echo "FOLD NUMBER: $fold"
    python ./src/models/yolov5/val.py --weights ./models/yolov5/batch16/yolov5x-img-512-fold-${fold}/weights/best.pt \
                                        --data src/models/yolov5/data/data_fold_${fold}.yaml \
                                        --img 512 \
                                        --conf-thres 0.143 \
                                        --iou-thres 0.5 \
                                        --project ./models/yolov5 \
                                        --name infer_fold_${fold} \
                                        --save-txt \
                                        --save-conf \
                                        --verbose 
    
    echo "###########################################################################################\n"
    sleep 3;
done

#                                         --max-det 3 \
#./input/dataset_folds_${fold}/images/valid \
#./input/tmp/covid/images/valid \



    # python ./src/models/yolov5/val.py --weights ./models/yolov5/yolov5x-img-512-fold-${fold}/weights/best.pt \
    #                                     --source ./input/dataset_folds_${fold}/images/valid \
    #                                     --img 512 \
    #                                     --conf 0.269 \
    #                                     --iou-thres 0.5 \
    #                                     --project ./models/yolov5 \
    #                                     --name infer_fold_${fold} \
    #                                     --save-txt \
    #                                     --save-conf \
