# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import sys; 

package_paths = [
#     '../input/pytorch-image-models/pytorch-image-models-master', #导入pytorch模型
    '../input/image-fmix/FMix-master'                            #FMix是一种数据增强方法（最近比较火的一种）
]

for pth in package_paths:
    sys.path.append(pth)


# %%
from glob import glob
import torch
import os
import random
import cv2
import pandas as pd
import numpy as np



import warnings
from sklearn.model_selection import GroupKFold, StratifiedKFold

# %% [markdown]
# # 1 数据预处理

# %%
# 将训练csv读入
# COMPETITION_NAME = "siimcovid19-512-img-png-600-study-png"
# load_dir = f"../input/{COMPETITION_NAME}/"
# df = pd.read_csv('../input/train_v1.csv')#'../input/siim-covid19-detection/train_study_level.csv')
# df.head()


# %%
# 为操作方便修改表头 inplace参数决定是否修改原df
# df.rename(columns={'Negative for Pneumonia':'0','Typical Appearance':'1',"Indeterminate Appearance":'2',
#                    "Atypical Appearance":"3"}, inplace=True)
# df.head()


# %%
# 解码one-hot
# labels = []
# def get_label(row):
#     for c in df.columns:
#         if row[c]==1:
#             labels.append(int(c))
# df.apply(get_label, axis=1)
# print("label modified")


# %%
# 合并两份DataFrame,注意axis = 1参数
# labels = {'label':labels}
# study_label = pd.DataFrame(labels)
# train_study = pd.concat([df, study_label], axis = 1)
#print(train_study)


# %%
# del train_study ['0'];del train_study ['1'];del train_study ['2'];del train_study ['3']
# train_study

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


CFG = {
    'fold_num': 5,
    'seed': 719,
    #'model_arch': 'tf_efficientnet_b7',
    'model_arch': 'tf_efficientnet_b4_ns',
    'img_size': 512,
    'epochs': 50,
    'train_bs': 14,
    'valid_bs': 32,
    'T_0': 10,
    'lr': 1e-6,
    'min_lr': 1e-6,
    'weight_decay':1e-6,
    'num_workers': 4,
    'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    'verbose_step': 1,
    'device': 'cuda:0'
}


if __name__ == '__main__':
    seed_everything(CFG['seed'])
    
    train_val_df = pd.read_csv('../input/train_classification_v1.csv')#'../input/siim-covid19-detection/train_study_level.csv')
    train_val_df.class_id.value_counts()
    
    folds = StratifiedKFold(n_splits=CFG['fold_num'], shuffle=True, random_state=CFG['seed']).split(np.arange(train_val_df.shape[0]), train_val_df.class_id.values)

    fold_df = train_val_df.copy()
    fold_df.loc[:, 'fold'] = 0
    
    for fold, (trn_idx, val_idx) in enumerate(folds):
        fold_df.loc[fold_df.iloc[val_idx].index, 'fold'] = fold
    print(f'Each Fold Value Counts')
    print(f'{fold_df.fold.value_counts()}')
    print(f'{fold_df.head()}')

    for fold in range(CFG['fold_num']):
        print(f'----------- Fold {fold} -----------------')
        print(fold_df[fold_df['fold']==fold]['class_id'].value_counts())

    if fold_df.isnull().sum().sum() > 0:
        # Count the NaN under an entire DataFrame:
        print(f'Number of NaN in DataFrame : {fold_df.isnull().sum().sum()}')
        sys.exit(1)
    else:
        fold_df.to_csv('../input/train_classification_fold_v1.csv', index=False)