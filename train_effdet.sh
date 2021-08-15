# train 4 class
# python -m src.train_efficientdet model=classification/efficientnet \
#                                     optimizer=adam \
#                                     scheduler=cosinewarm \
#                                     loss=cross_entropy \
#                                     +experiment=detection_effdet \
#                                     model_id=detection_tf_efficientnetv2_l \
#                                     data.n_fold=5 \
#                                     train.monitor=val_score \
#                                     train.mode=max \
#                                     system.gpus=\"0\"


# python -m src.train_efficientdet model=detection/efficientdet \
#                                     optimizer=adamw \
#                                     scheduler=cosinewarm \
#                                     loss=cross_entropy \
#                                     +experiment=detection_effdet \
#                                     model_id=detection_tf_efficientnetv2_l \
                                    # model.base_name=efficientnetv2_l_in21ft1k
#                                     model.model_name=tf_$\{model.base_name\} \
#                                     data.n_fold=5 \
#                                     train.monitor=valid_loss \
#                                     train.mode=min \
#                                     train.batch_size=2 \
#                                     system.gpus=\"0\"

#                                    model.model_name=tf_$\{model.base_name\}_m_in21ft1k \


# python -m src.train_efficientdet model=classification/efficientnet_v2 \
#                                     optimizer=adam \
#                                     scheduler=cosinewarm \
#                                     loss=cross_entropy \
#                                     +experiment=detection_effdet \
#                                     model_id=detection_tf_efficientnetv2_l \
#                                     data.n_fold=5 \
#                                     model.model_name=tf_$\{model.base_name\}_m_in21ft1k \
#                                     train.monitor=val_score \
#                                     train.mode=max \
#                                     train.batch_size=12 \
#                                     system.gpus=\"0\"


python -m src.train_efficientdet model=detection/efficientdet \
                                    optimizer=adamw \
                                    scheduler=cosinewarm \
                                    loss=cross_entropy \
                                    +experiment=detection_effdet \
                                    model_id=detection_tf_efficientnet_b7 \
                                    model.base_name=efficientnet_b7 \
                                    model.model_name=tf_$\{model.base_name\} \
                                    model.input_size=768 \
                                    model.output_size=768 \
                                    data.n_fold=5 \
                                    train.monitor=valid_loss \
                                    train.mode=min \
                                    train.batch_size=2 \
                                    system.gpus=\"0\"