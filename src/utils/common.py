from typing import Any
import importlib
import random
import os
import numpy as np
import torch

import tensorboard
from glob import glob
from omegaconf import DictConfig
from pytorch_lightning.callbacks import (EarlyStopping,
                                         ModelCheckpoint)
from pytorch_lightning.loggers import CometLogger


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_obj(obj_path: str, default_obj_path: str = '') -> Any:
    """
    Extract an object from a given path.
    https://github.com/quantumblacklabs/kedro/blob/9809bd7ca0556531fa4a2fc02d5b2dc26cf8fa97/kedro/utils.py
        Args:
            obj_path: Path to an object to be extracted, including the object name.
            default_obj_path: Default object path.
        Returns:
            Extracted object.
        Raises:
            AttributeError: When the object does not have the given named attribute.
    """
    obj_path_list = obj_path.rsplit('.', 1)
    obj_path = obj_path_list.pop(0) if len(obj_path_list) > 1 else default_obj_path
    obj_name = obj_path_list[0]
    module_obj = importlib.import_module(obj_path)
    if not hasattr(module_obj, obj_name):
        raise AttributeError(f'Object `{obj_name}` cannot be loaded from `{obj_path}`.')
    return getattr(module_obj, obj_name)


def save_model(cfg: DictConfig, ckpt_dir:str, fold:int) -> None:
    """実験用モデル(訓練済み)を重みのみ保存する。

    Args:
        cfg (CfgNode): config
        ckpt_dir (str): 訓練済みモデルが保存されているディレクトリ
        fold: 対象のfold
        
    Returns:
        None
    """
    ckpt_path = glob(ckpt_dir + f'/fold{fold}*.ckpt')

    if len(ckpt_path) == 1:
        ckpt_path = ckpt_path[0]
        print(f'Found Pretrained Weight : {ckpt_path}')
    elif len(ckpt_path) > 1:
        print(f'There are more than one weight file found : {ckpt_path}')
    else:
        print(f'Weight file not found : {ckpt_path}')
        
    if cfg.task == 'detection':
        from src.train_efficientdet import EfficientDetModel
        model = EfficientDetModel.load_from_checkpoint(checkpoint_path=str(ckpt_path), cfg=cfg, fold=fold, pretrained_weights=False)

        # save as a simple torch model
        ckpt_model_name = ckpt_path.replace('ckpt','pth')

        print(f'Ckpt Model saved to : {ckpt_model_name}')
        torch.save(model.model.state_dict(), ckpt_model_name)


    
def get_callback(cfg:DictConfig, output_path:str, fold:int) -> Any:
    """実験用コールバック関数(src/tools/train.pyで使用)を返す。

    Args:
        cfg (CfgNode): config
        output_path (str): 出力先のパス
        fold (int): fold番号
        
    Returns:
        logger (pytorch_lightning.loggers): pytorch lightningで用意されているLogging関数(自作も可)
        checkpoint (ModelCheckpoint): pytorch lightningにおけるモデルの保存設定
    """
    
    loggers = []
    if cfg.callback.logger.comet.flag:
        print(f'Comet Logger: {cfg.callback.logger.comet.flag}')
        comet_logger = CometLogger(save_dir=cfg.callback.logger.comet.save_dir,
                                workspace=cfg.callback.logger.comet.workspace,
                                project_name=cfg.callback.logger.comet.project_name,
                                api_key=cfg.private.comet_api,
                                experiment_name=os.getcwd().split('\\')[-1])
        loggers.append(comet_logger)

    # tb_logger = TensorBoardLogger(save_dir=output_path)
    # loggers.append(tb_logger)

    # lr_logger = LearningRateLogger()

    monitor_name = cfg.callback.model_checkpoint.params.monitor
    if monitor_name == 'valid_loss':
        model_checkpoint = ModelCheckpoint(dirpath=output_path, filename=f'fold{fold}' + '_{epoch}_{valid_loss:.3f}',
                                       **cfg.callback.model_checkpoint.params)
    elif monitor_name == 'val_score':
        model_checkpoint = ModelCheckpoint(dirpath=output_path, filename='fold{fold}' + '_{epoch}_{val_score:.3f}',
                                       **cfg.callback.model_checkpoint.params)
    else:
        model_checkpoint = ModelCheckpoint(dirpath=output_path, filename=f'fold{fold}' + '_{epoch}_{other_metric:.3f}',
                                        **cfg.callback.model_checkpoint.params)
    
    print(f'Early stopping: {cfg.callback.early_stopping.flag}')                                   
    if cfg.callback.early_stopping.flag:
        early_stopping = EarlyStopping(**cfg.callback.early_stopping.params)
    else:
        early_stopping = False

    return loggers, model_checkpoint, early_stopping


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0