import sys
from typing import Dict, Optional, Tuple

import albumentations as A
import cv2
import numpy as np
import omegaconf
import torch
import hydra

from src.classification.fmix import binarise_mask, make_low_freq_image, sample_mask
from hydra.utils import instantiate
from omegaconf import DictConfig
# from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset

from src.utils.common import load_obj


def get_img(path):
    im_bgr = cv2.imread(path)
    if im_bgr is None:
        print('no image', path)
    im_rgb = im_bgr[:, :, ::-1]
    #return im_bgr
    return im_rgb


def rand_bbox(size, lam):
    W = size[0]
    H = size[1]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2


class CustomDataset(Dataset):
    def __init__(self, cfg, train_df, data_root, 
                 transforms=None, 
                 output_label=True, 
                 one_hot_label=False,
                 do_fmix=False, 
                 fmix_params={
                     'alpha': 1., 
                     'decay_power': 3., 
                     'shape': (512, 512),
                     'max_soft': True, 
                     'reformulate': False
                 },
                 do_cutmix=False,
                 cutmix_params={
                     'alpha': 1,
                 }
                ):
        
        super().__init__()
        self.cfg = cfg
        self.df = train_df.reset_index(drop=True).copy()
        if 'study_id' in self.df.columns:
            self.id_col_name = 'study_id'
        else:
            self.id_col_name = 'id'
        self.transforms = transforms
        self.data_root = self.cfg.data.train_image_dir
        self.do_fmix = do_fmix
        self.fmix_params = fmix_params
        self.do_cutmix = do_cutmix
        self.cutmix_params = cutmix_params
        
        self.output_label = output_label
        self.one_hot_label = one_hot_label
        
        if output_label == True:
            self.labels = self.df['class_id'].values
            #print(self.labels)
            
            if one_hot_label is True:
                self.labels = np.eye(self.df['class_id'].max()+1)[self.labels]
                #print(self.labels)
            
    def __len__(self):
        return self.df.shape[0]
    
    def __getitem__(self, index: int):
        
        # get labels
        if self.output_label:
            target = self.labels[index]
        # img  = get_img("{}/{}".format(self.data_root, self.df.loc[index][self.id_col_name])+'.png')
        img_name = self.data_root + f'/{self.df.loc[index][self.id_col_name]+".png"}'
        img  = get_img(hydra.utils.to_absolute_path(img_name))

        if self.transforms:
            img = self.transforms(image=img)['image']
        
        if self.do_fmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            with torch.no_grad():
                #lam, mask = sample_mask(**self.fmix_params)
                
                lam = np.clip(np.random.beta(self.fmix_params['alpha'], self.fmix_params['alpha']),0.6,0.7)
                
                # Make mask, get mean / std
                mask = make_low_freq_image(self.fmix_params['decay_power'], self.fmix_params['shape'])
                mask = binarise_mask(mask, lam, self.fmix_params['shape'], self.fmix_params['max_soft'])
    
                fmix_ix = np.random.choice(self.df.index, size=1)[0]
                fmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[fmix_ix][self.id_col_name])+'.png')

                if self.transforms:
                    fmix_img = self.transforms(image=fmix_img)['image']

                mask_torch = torch.from_numpy(mask)
                
                # mix image
                img = mask_torch*img+(1.-mask_torch)*fmix_img

                #print(mask.shape)

                #assert self.output_label==True and self.one_hot_label==True

                # mix target
                rate = mask.sum()/self.cfg.model.input_size.width/self.cfg.model.input_size.height
                target = rate*target + (1.-rate)*self.labels[fmix_ix]
                #print(target, mask, img)
                #assert False
        
        if self.do_cutmix and np.random.uniform(0., 1., size=1)[0] > 0.5:
            #print(img.sum(), img.shape)
            with torch.no_grad():
                cmix_ix = np.random.choice(self.df.index, size=1)[0]
                cmix_img  = get_img("{}/{}".format(self.data_root, self.df.iloc[cmix_ix][self.id_col_name])+'.png')
                if self.transforms:
                    cmix_img = self.transforms(image=cmix_img)['image']
                    
                lam = np.clip(np.random.beta(self.cutmix_params['alpha'], self.cutmix_params['alpha']),0.3,0.4)
                bbx1, bby1, bbx2, bby2 = rand_bbox((self.cfg.model.input_size.width, self.cfg.model.input_size.height), lam)

                img[:, bbx1:bbx2, bby1:bby2] = cmix_img[:, bbx1:bbx2, bby1:bby2]

                rate = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (self.cfg.model.input_size.width * self.cfg.model.input_size.height))
                target = rate*target + (1.-rate)*self.labels[cmix_ix]
                
            #print('-', img.sum())
            #print(target)
            #assert False
                            
        # do label smoothing
        #print(type(img), type(target))
        if self.output_label == True:
            return img, target
        else:
            return img

def load_augs(cfg: DictConfig, bbox_params: Optional[DictConfig] = None) -> A.Compose:
    """
    Load albumentations
    Args:
        cfg: model config
        bbox_params: bbox parameters
    Returns:
        composed object
    """
    augs = []
    for a in cfg:
        if a['class_name'] == 'albumentations.OneOf':
            small_augs = []
            for small_aug in a['params']:
                # yaml can't contain tuples, so we need to convert manually
                params = {k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v)) for k, v in
                          small_aug['params'].items()}
                aug = load_obj(small_aug['class_name'])(**params)
                small_augs.append(aug)
            aug = load_obj(a['class_name'])(small_augs)
            aug.p=a['p']
            augs.append(aug)
        else:
            params = {k: (v if type(v) != omegaconf.listconfig.ListConfig else tuple(v)) for k, v in
                      a['params'].items()}
            aug = load_obj(a['class_name'])(**params)
            augs.append(aug)
    if bbox_params is not None:
        transforms = A.Compose(augs, bbox_params=instantiate(bbox_params))
    else:
        transforms = A.Compose(augs)
    return transforms
