# train 4 class
# python -m src.train_classification model=classification/efficientnet \
#                                     optimizer=adam \
#                                     scheduler=cosinewarm \
#                                     loss=cross_entropy \
#                                     +experiment=classification_study \
#                                     model_id=study_cls_tf_efficientnetb4_ns \
#                                     data.n_fold=5 \
#                                     train.monitor=val_score \
#                                     train.mode=max \
#                                     system.gpus=\"0\"


python -m src.train_classification model=classification/efficientnet_v2 \
                                    optimizer=adam \
                                    scheduler=cosinewarm \
                                    loss=cross_entropy \
                                    +experiment=classification_study \
                                    model_id=study_cls_tf_efficientnet_v2_l \
                                    data.n_fold=5 \
                                    train.monitor=val_score \
                                    train.mode=max \
                                    train.batch_size=6 \
                                    system.gpus=\"0\"


python -m src.train_classification model=classification/efficientnet_v2 \
                                    optimizer=adam \
                                    scheduler=cosinewarm \
                                    loss=cross_entropy \
                                    +experiment=classification_study \
                                    model_id=study_cls_tf_efficientnet_v2_m \
                                    data.n_fold=5 \
                                    model.model_name=tf_$\{model.base_name\}_m_in21ft1k \
                                    train.monitor=val_score \
                                    train.mode=max \
                                    train.batch_size=12 \
                                    system.gpus=\"0\"