# train 2 class
# python -m src.train_classification model=classification/efficientnet \
#                                     optimizer=adam \
#                                     scheduler=cosinewarm \
#                                     loss=cross_entropy \
#                                     +experiment=classification_2class \
#                                     model_id=binary_cls_tf_efficientnetb4_ns \
#                                     data.n_fold=5 \
#                                     train.monitor=val_score \
#                                     train.mode=max \
#                                     system.gpus=\"0\"


python -m src.train_classification model=classification/efficientnet_v2 \
                                    optimizer=adam \
                                    scheduler=cosinewarm \
                                    loss=cross_entropy \
                                    +experiment=classification_2class \
                                    model_id=binary_cls_tf_efficientnet_v2_l \
                                    data.n_fold=5 \
                                    train.monitor=val_score \
                                    train.batch_size=6 \
                                    train.mode=max \
                                    system.gpus=\"0\"