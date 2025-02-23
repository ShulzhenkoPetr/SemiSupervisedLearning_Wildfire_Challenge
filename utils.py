import os
import yaml
from pathlib import Path
from typing import List, Tuple, Dict
from types import SimpleNamespace
from functools import partial
import numpy as np
import torch
from torch import nn

from models.models import NaiveNet, CustomResNet


def load_config(config_path: str) -> SimpleNamespace:
    """Reads a YAML config and returns the content as a dict."""
    try:
        with open(config_path, 'r') as file:
            data = yaml.safe_load(file)
            dot_dict = SimpleNamespace(**data)
            return dot_dict
    except FileNotFoundError:
        print(f"Error: The file '{config_path}' was not found.")
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")


def set_seed(seed: int = 42):
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def precision_fn(tp, fp, eps=1e-8):
    return tp / (tp + fp + eps)

def recall_fn(tp, fn, eps=1e-8):
    return tp / (tp + fn + eps)

def f1_score_fn(precision, recall , eps=1e-8):
    return 2 * precision * recall / (precision + recall + eps)


def acc_fn(preds, targets):
    return torch.sum(preds == targets) / preds.shape[0]

def acc_fn_from_logits_binary(preds, targets, th=0.5) -> float:
    preds = torch.sigmoid(preds)
    return acc_fn(preds >= th, targets)


def compute_metric(pred_labels, targets, metric: str):
    if metric == 'tp':
        return torch.sum(pred_labels == targets)
    elif metric == 'fp': 
        return torch.sum((pred_labels != targets) & (pred_labels == 1))
    elif metric == 'fn':
        return torch.sum((pred_labels != targets) & (pred_labels == 0))
    else:
        raise ValueError("Invalid metric type. Use 'tp', 'fp', or 'fn'.")

def compute_metric_from_logits(preds, targets, metric: str, th=0.5):
    pred_labels = torch.sigmoid(preds) >= th 
    return compute_metric(pred_labels, targets, metric)

def get_metric_fn(metric_name):
    if 'CE' in metric_name:
        return nn.BCEWithLogitsLoss(reduction='mean')
    elif 'Acc' in metric_name:
        return acc_fn_from_logits_binary
    elif 'tp' in metric_name:
        return partial(compute_metric_from_logits, metric='tp')
    elif 'fp' in metric_name:
        return partial(compute_metric_from_logits, metric='fp')
    elif 'fn' in metric_name:
        return partial(compute_metric_from_logits, metric='fn')
    elif 'precision' in metric_name or 'recall' in metric_name or 'f1' in metric_name: # skip to calculate globally
        return None
    else:
        raise NotImplementedError(f'{metric_name}')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, logger):
        self.name = name
        self.logger = logger
        self.val, self.avg, self.sum, self.count = 0, 0, 0, 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def log_average(self, epoch: int = None):
        self.logger.log({self.name: self.avg}) #, step=epoch, commit=True
    
    def log_sum(self, epoch: int = None):
        self.logger.log({self.name: self.sum})


class ModelCheckpoint:

    def __init__(self, 
                 monitor: str, 
                 output_path: Path, 
                 top_k: int = 1,
                 mode: str = 'max',
                 config: SimpleNamespace = None) -> None:
        self.monitor = monitor
        self.top_k = top_k
        if mode != 'max' and mode != 'min':
            raise NotImplementedError
        self.mode = mode

        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.output_path = output_path

        self.config = config

        self.ckpt_dict = {}

    def update(self, metrics_dict: Dict, model: nn.Module, optimizer: torch.optim.Optimizer, epoch: int) -> None:
        metric_value = metrics_dict[self.monitor]

        if len(self.ckpt_dict) < self.top_k or (
            (self.mode == 'max' and metric_value > min(self.ckpt_dict.keys())) or 
            (self.mode == 'min' and metric_value < max(self.ckpt_dict.keys()))
        ):
            if len(self.ckpt_dict) >= self.top_k and self.mode == 'max':
                del self.ckpt_dict[min(self.ckpt_dict.keys())]

            if len(self.ckpt_dict) >= self.top_k and self.mode == 'min':
                del self.ckpt_dict[max(self.ckpt_dict.keys())]

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            self.ckpt_dict[metric_value] = checkpoint

    def save(self) -> None:
        for d in self.ckpt_dict.values():
            torch.save(d, self.output_path / f'checkpoint_epoch_{d["epoch"]}.pth')
        
        if self.config:
            config_dict = vars(self.config)
            with open(self.output_path / 'config.yaml', 'w') as f:
                yaml.dump(config_dict, f, default_flow_style=False)


def load_checkpoint(path2ckpt: Path, config: SimpleNamespace):
    checkpoint_dict = torch.load(path2ckpt)
    if config.model_name == 'CustomResNet':
        model = CustomResNet(config)
    elif config.model_name == 'NaiveNet':
        model = NaiveNet()
    else:
        raise NotImplementedError(f'This model {config.model.name} is not implemented')
    
    model.load_state_dict(checkpoint_dict['model_state_dict'])

    return model