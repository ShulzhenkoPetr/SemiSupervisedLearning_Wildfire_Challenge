from typing import Any, Dict
import argparse
import json
from types import SimpleNamespace
from functools import partial
from pathlib import Path
import wandb
from tqdm import tqdm

import torch

from utils import load_config, set_seed
from models.models import CustomResNet, NaiveNet, TTAWrapper, MajorityVotingEnsemble, load_checkpoint
from dataset_utils import get_val_transforms, get_dataloader
from train import evaluate


def test_model(path2ckpt: Path, config: SimpleNamespace, model_type: str = 'base') -> None:
    set_seed(27)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if model_type == 'base':
        model = load_checkpoint(path2ckpt, config)
    elif model_type == 'tta':
        model = TTAWrapper(base_model=load_checkpoint(path2ckpt, config))
    elif model_type == 'ensemble':
        model = MajorityVotingEnsemble(
            [
                # (Path('checkpoints/medium_augs_unlabelled_norms_naiveNet/checkpoint_epoch_21.pth'), 
                #  load_config('checkpoints/medium_augs_unlabelled_norms_naiveNet/config.yaml')),
                # (Path('checkpoints/unlabelled_norms_naiveNet/checkpoint_epoch_29.pth'),
                #  load_config('checkpoints/medium_augs_unlabelled_norms_naiveNet/config.yaml')),
                (Path('checkpoints/CustomResNet_medium_augs_rot0_nojitter_unlabelled_norms_warmup15_scheduler_epochsNone/checkpoint_epoch_28.pth'),
                 load_config('checkpoints/CustomResNet_medium_augs_rot0_nojitter_unlabelled_norms_warmup15_scheduler_epochsNone/config.yaml')),
                 (Path('checkpoints/CustomResNet_medium_augs_rot30_nojitter_unlabelled_norms_warmup15_scheduler_epochsNone/checkpoint_epoch_28.pth'),
                  load_config('checkpoints/CustomResNet_medium_augs_rot30_nojitter_unlabelled_norms_warmup15_scheduler_epochsNone/config.yaml'))
            ],
            device=device
        )
    else:
        raise NotImplementedError(f'{model_type}')
    
    model.to(device)

    val_transforms = get_val_transforms(dataset_stats=config.dataset_stats)
    test_dataloader = get_dataloader(
        folder_path=Path(config.test_data_path), transforms=val_transforms, batch_size=config.batch_size, shuffle=False
    )

    test_metrics_dict = evaluate(
        model=model,
        dataloader=test_dataloader,
        config=config,
        logger=None, epoch=None, 
        device=device, phase_type='test',
        is_from_logits=model_type=='base'
    )
    test_metrics_dict = {name: val.item() for name, val in test_metrics_dict.items()}
    print('Test results: ')
    for name, val in test_metrics_dict.items():
        print(f'{name} = {round(val, 3)}')

    if model_type != 'ensemble':
        with open(path2ckpt.with_name(f'{model_type}_test_metrics.json'), 'w') as f:
            json.dump(test_metrics_dict, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test config parser")
    parser.add_argument("-pconfig", "--path2config", type=str, 
                        help="The path to yaml config file")
    parser.add_argument("-pckpt", "--path2checkpoint", type=str,
                        help="The path to model checkpoint file")
    parser.add_argument("-mtype", "--modeltype", type=str,
                        help="The type of model: base, tta or ensemble")
    args = parser.parse_args()

    config = load_config(args.path2config)
    if args.modeltype == 'ensemble':
        test_model(None, config, 'ensemble')
    else:
        test_model(Path(args.path2checkpoint), config, args.modeltype)
