import sys

package_paths = [
#     '../input/pytorch-image-models/pytorch-image-models-master', #导入pytorch模型
    '../input/image-fmix/FMix-master'                            #FMix是一种数据增强方法（最近比较火的一种）
]

for pth in package_paths:
    sys.path.append(pth)

import os
import random
import time
import warnings
from glob import glob
import hydra
import logging
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from catalyst.data.sampler import BalanceClassSampler
from src.utils.common import load_obj
from src.classification.data import CustomDataset, load_augs
from src.classification.models import ImgClassifier
from src.utils.tools import EarlyStopping
from src.utils.common import seed_everything


def prepare_dataloader(cfg, log, train_, valid_, data_root='../input/siimcovid19-512-img-png-600-study-png/study'):
    
    # train_ = df.loc[trn_idx,:].reset_index(drop=True)
    # valid_ = df.loc[val_idx,:].reset_index(drop=True)
    # print(train_, valid_)
    # print(cfg)

    # if 'albumentations_classification' in cfg.augmentaion.framework:
    log.info('Applying Albumentations Classification')
    train_augs = load_augs(cfg['albumentations']['train']['augs'])
    valid_augs = load_augs(cfg['albumentations']['valid']['augs'])
        
    train_ds = CustomDataset(cfg, train_, data_root, transforms=train_augs, output_label=True, one_hot_label=False, do_fmix=False, do_cutmix=False)
    valid_ds = CustomDataset(cfg, valid_, data_root, transforms=valid_augs, output_label=True)
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        pin_memory=False,
        drop_last=False,
        shuffle=True,        
        num_workers=cfg.system.num_workers,
        #sampler=BalanceClassSampler(labels=train_['label'].values, mode="downsampling")
    )
    val_loader = torch.utils.data.DataLoader(
        valid_ds, 
        batch_size=cfg.train.val_batch_size,
        num_workers=cfg.system.num_workers,
        shuffle=False,
        pin_memory=False,
    )
    return train_loader, val_loader


def train_one_epoch(cfg, log, epoch, model, loss_fn, optimizer, train_loader, device, scheduler=None, schd_batch_update=False, verbose_step=1, scaler=None):
    #print(epoch, model, loss_fn, optimizer, train_loader, device)
    model.train()

    t = time.time()
    running_loss = None

    pbar = tqdm(enumerate(train_loader), total=len(train_loader))
    #print(enumerate(train_loader),'len(train_loader)',train_loader)
    
    for step, (imgs, image_labels) in pbar:
        
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()

        #print(image_labels.shape, exam_label.shape)
        with autocast():
            image_preds = model(imgs)   #output = model(input)
            #print(image_preds.shape, exam_pred.shape)

            loss = loss_fn(image_preds, image_labels)
            
            scaler.scale(loss).backward()

            if running_loss is None:
                running_loss = loss.item()
            else:
                running_loss = running_loss * .99 + loss.item() * .01

            if ((step + 1) %  cfg.trainer.accumulate_grad_batches == 0) or ((step + 1) == len(train_loader)):
                # may unscale_ here if desired (e.g., to allow clipping unscaled gradients)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad() 
                
                if scheduler is not None and schd_batch_update:
                    scheduler.step()

            if ((step + 1) % verbose_step == 0) or ((step + 1) == len(train_loader)):
                description = f'epoch {epoch} loss: {running_loss:.4f}'
                
                pbar.set_description(description)
                
    if scheduler is not None and not schd_batch_update:
        scheduler.step()


def valid_one_epoch(cfg, log, epoch, model, loss_fn, val_loader, device, scheduler=None, schd_loss_update=False, verbose_step=1):
    model.eval()

    t = time.time()
    loss_sum = 0
    sample_num = 0
    image_preds_all = []
    image_targets_all = []
    
    pbar = tqdm(enumerate(val_loader), total=len(val_loader))
    for step, (imgs, image_labels) in pbar:
        imgs = imgs.to(device).float()
        image_labels = image_labels.to(device).long()
        
        image_preds = model(imgs)   #output = model(input)
        #print(image_preds.shape, exam_pred.shape)
        image_preds_all += [torch.argmax(image_preds, 1).detach().cpu().numpy()]
        image_targets_all += [image_labels.detach().cpu().numpy()]
        
        loss = loss_fn(image_preds, image_labels)
        
        loss_sum += loss.item()*image_labels.shape[0]
        sample_num += image_labels.shape[0]  

        if ((step + 1) % verbose_step == 0) or ((step + 1) == len(val_loader)):
            description = f'epoch {epoch} loss: {loss_sum/sample_num:.4f}'
            pbar.set_description(description)
    
    image_preds_all = np.concatenate(image_preds_all)
    image_targets_all = np.concatenate(image_targets_all)
    log.info('validation multi-class accuracy = {:.4f}'.format((image_preds_all==image_targets_all).mean()))
    val_acc = (image_preds_all==image_targets_all).mean()
    if scheduler is not None:
        if schd_loss_update:
            scheduler.step(loss_sum/sample_num)
        else:
            scheduler.step()
    return val_acc


@hydra.main(config_path="../configs", config_name="train_classification")
def main(cfg):
    log = logging.getLogger(__name__)
    # cfg = {
    #     'fold_num': 5,
    #     'seed': 719,
    #     #'model_arch': 'tf_efficientnet_b7',
    #     'model_arch': 'tf_efficientnet_b4_ns',
    #     'img_size': 512,
    #     'epochs': 100,
    #     'train_bs': 14,
    #     'valid_bs': 32,
    #     'T_0': 10,
    #     'lr': 1e-6,
    #     'min_lr': 1e-6,
    #     'weight_decay':1e-6,
    #     'num_workers': 4,
    #     'accum_iter': 2, # suppoprt to do batch accumulation for backprop with effectively larger batch size
    #     'verbose_step': 1,
    #     'device': 'cuda:0'
    # }
     # for training only, need nightly build pytorch
    train_img_path = cfg.data.train_image_dir #'../input/siimcovid19-512-img-png-600-study-png/study'
    # cfg.data.csv_path = '../input/train_classification_fold_v1.csv'
    model_path = f'./weights/' #models/{exp_id}/'
    os.makedirs(model_path, exist_ok=True)
    seed_everything(cfg.system.seed)
    
    train = pd.read_csv(hydra.utils.to_absolute_path(cfg.data.csv_path))
    
    for idx, fold in enumerate(range(cfg.data.n_fold)):
        # if fold < 4:
        #     continue
        train_df = train[train['fold'] != fold]
        valid_df = train[train['fold'] == fold]
        best_score = 0.0

        log.info('------------ Training with {} started ----------------'.format(fold))

        patience = 15
        early_stopping = EarlyStopping(patience=patience, verbose=True, mode='max') 

        train_loader, val_loader = prepare_dataloader(cfg, log, train_df, valid_df, data_root=train_img_path)

        if cfg.system.device == 'GPU':
            device = torch.device('cuda:0')

        assert train.class_id.nunique() == cfg.data.num_classes
        model = ImgClassifier(cfg.model.model_name, cfg.data.num_classes, pretrained=True).to(device)
        scaler = GradScaler()
        # optimizer = torch.optim.Adam(model.parameters(), lr=cfg.optimizer.params.lr, weight_decay=cfg.optimizer.params.weight_decay)
        #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, gamma=0.1, step_size=cfg['epochs']-1)
        # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=cfg['T_0'], T_mult=1, eta_min=cfg['min_lr'], last_epoch=-1)
        optimizer = load_obj(cfg.optimizer.class_name)(model.parameters(), **cfg.optimizer.params)
        scheduler = load_obj(cfg.scheduler.class_name)(optimizer, **cfg.scheduler.params)
        #scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=25, 
        #                                                max_lr=cfg['lr'], epochs=cfg['epochs'], steps_per_epoch=len(train_loader))
        criterion = load_obj(cfg.loss.class_name)(**cfg.loss.params)
        
        for epoch in range(cfg.train.epochs):
            train_one_epoch(cfg, log, epoch, model, criterion, optimizer, train_loader, device, scheduler=scheduler, schd_batch_update=False, scaler=scaler)

            with torch.no_grad():
                val_acc = valid_one_epoch(cfg, log, epoch, model, criterion, val_loader, device, scheduler=None, schd_loss_update=False)
                # early_stopping needs the validation loss to check if it has decresed, 
                # and if it has, it will make a checkpoint of the current model
                early_stopping(val_acc, model)
                
                if early_stopping.early_stop:
                    print("Early stopping")
                    break
            if val_acc > best_score:
                log.info(f'Best Multi-Class Accuracy: {val_acc}')
                best_score = val_acc
                torch.save(model.state_dict(),f'{model_path}best_acc_{cfg.model.model_name}_fold{fold}.pt')
            
        #torch.save(model.cnn_model.state_dict(),'{}/cnn_model_fold_{}_{}'.format(cfg['model_path'], fold, cfg['tag']))
        del model, optimizer, train_loader, val_loader, scaler, scheduler
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()