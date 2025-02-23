from typing import Any, Dict
import argparse
import json
from types import SimpleNamespace
from functools import partial
from pathlib import Path
import wandb
from tqdm import tqdm

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR

from utils import load_config, set_seed, ModelCheckpoint, AverageMeter, get_metric_fn, precision_fn, recall_fn, f1_score_fn
from models.models import CustomResNet, NaiveNet
from dataset_utils import get_val_transforms, get_dataloader
from train import evaluate


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


def test_model(path2ckpt: Path, config: SimpleNamespace) -> None:
    set_seed(27)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = load_checkpoint(path2ckpt, config)
    model.to(device)

    val_transforms = get_val_transforms(dataset_stats=config.dataset_stats)
    test_dataloader = get_dataloader(
        Path(config.test_data_path), transforms=val_transforms, batch_size=config.batch_size, shuffle=False
    )

    test_metrics_dict = evaluate(
        model=model,
        dataloader=test_dataloader,
        config=config,
        logger=None, epoch=None, 
        device=device, phase_type='test'
    )
    test_metrics_dict = {name: val.item() for name, val in test_metrics_dict.items()}
    print('Test results: ')
    for name, val in test_metrics_dict.items():
        print(f'{name} = {round(val, 3)}')

    with open(path2ckpt.with_name('test_metrics.json'), 'w') as f:
        json.dump(test_metrics_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Experiment config parser")
    parser.add_argument("-pconfig", "--path2config", type=str, 
                        help="The path to yaml config file")
    parser.add_argument("-pckpt", "--path2checkpoint", type=str,
                        help="The path to model checkpoint file")
    args = parser.parse_args()

    config = load_config(args.path2config)

    test_model(Path(args.path2checkpoint), config)
