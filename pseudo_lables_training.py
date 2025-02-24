from typing import Any, Dict
import argparse
from types import SimpleNamespace
from functools import partial
from pathlib import Path
import wandb
from tqdm import tqdm

from omegaconf import DictConfig, OmegaConf
import hydra

import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR, MultiStepLR, LambdaLR

from utils import load_config, set_seed, ModelCheckpoint, AverageMeter, get_metric_fn, precision_fn, recall_fn, f1_score_fn
from dataset_utils import get_train_transforms, get_val_transforms, get_dataloader, unlbl_collate_fn, \
                        PseudoLabelDataset, LabelledDataset
from train import train_epoch, evaluate
from models.models import load_checkpoint


def generate_pseudo_labels(
        pretrained_model: nn.Module, 
        unlabelled_dataloader: torch.utils.data.DataLoader,
        confidence_th: float,
        device: torch.device) -> Dict:
    
    path2label = {}
    pretrained_model.to(device)
    pretrained_model.eval()
    with torch.no_grad():
        for imgs, paths in tqdm(unlabelled_dataloader, desc='Pseudo-labels generation'):
            imgs = imgs.to(device)
            preds = pretrained_model(imgs)
            pred_probas = torch.sigmoid(preds)
            pred_labels = pred_probas >= 0.5

            for lbl, proba, path in zip(pred_labels, pred_probas.flatten(), paths):
                if abs(proba - 0.5) > confidence_th:
                    path2label[path] = lbl
    
    return path2label


def ssl_step(pretrained_model: nn.Module, config: SimpleNamespace, ssl_iter: int) -> nn.Module:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Generate pseudo-labels
    val_transforms = get_val_transforms(dataset_stats=config.dataset_stats)
    unlabelled_dataloader = get_dataloader(
        folder_path=Path(config.unlabelled_data_path),
        dataset_type='unlabelled',
        collate_fn=unlbl_collate_fn,
        transforms=val_transforms, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    path2label = generate_pseudo_labels(
        pretrained_model=pretrained_model,
        unlabelled_dataloader=unlabelled_dataloader,
        confidence_th=config.confidence_th,
        device=device
    )

    print(f'SSL iter {ssl_iter}: generated {len(path2label.keys())} pseudo-labels')

    # Setup dataloader for training
    train_transforms = get_train_transforms(config=config, dataset_stats=config.dataset_stats)
    pseudo_lbl_dataset = PseudoLabelDataset(path2label=path2label, transforms=train_transforms)
    lbl_dataset = LabelledDataset(folder_path=Path(config.train_data_path), transforms=train_transforms)
    
    augmented_dataloader = get_dataloader(
        datasets=[lbl_dataset, pseudo_lbl_dataset],
        batch_size=config.batch_size, 
        shuffle=True
    )

    val_dataloader = get_dataloader(
        folder_path=Path(config.val_data_path), transforms=val_transforms, batch_size=config.batch_size, shuffle=False
    )

    pretrained_model.to(device)

    # Setup training

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    if config.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(
            params=pretrained_model.parameters(), 
            lr=float(config.lr) / ssl_iter,
            betas=(0.9, 0.999),
            eps=1e-5,
            weight_decay=float(config.weight_decay)
        )
    def warmup_lr(batch_i, nbr_warmup_batches=300):
        if batch_i < nbr_warmup_batches:
            return (batch_i + 1) / nbr_warmup_batches
        return 1.0
    if config.nbr_warmup_batches:
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=partial(warmup_lr, nbr_warmup_batches=config.nbr_warmup_batches))

    saving_path = Path(config.saving_path)
    saving_path = saving_path / f'{config.exp_name}_ssl_iter{ssl_iter}'
    model_checkpointer = ModelCheckpoint(
        monitor=config.monitor_metric, 
        output_path=saving_path, 
        top_k=config.k_ckpts, 
        mode=config.ckpt_mode,
        config=config
    )

    logger = wandb.init(
        project="wildfire",
        # entity="petr-shuzlhenko",
        name=f'{config.exp_name}_ssl_iter{ssl_iter}'     
    )
    val_metrics_dict = evaluate(pretrained_model, 
                                dataloader=val_dataloader, 
                                config=config, 
                                logger=logger, 
                                epoch=0, 
                                device=device,
                                phase_type='val',)
    # Launch training
    for epoch in tqdm(range(config.num_epochs), desc=f'Train {config.num_epochs} epochs'):
        train_epoch(pretrained_model, 
                    train_dataloader=augmented_dataloader, 
                    criterion=criterion, 
                    optimizer=optimizer,  
                    logger=logger, config=config, 
                    epoch=epoch, device=device,
                    warmup_scheduler=warmup_scheduler)
        if val_dataloader is not None:
            val_metrics_dict = evaluate(pretrained_model, 
                                        dataloader=val_dataloader, 
                                        config=config, 
                                        logger=logger, 
                                        epoch=epoch, 
                                        device=device,
                                        phase_type='val',)

        model_checkpointer.update(val_metrics_dict, pretrained_model, optimizer, epoch)
        
    model_checkpointer.save()

    return pretrained_model


def ssl_training(config: SimpleNamespace) -> None:

    model = load_checkpoint(config.path2ckpt, config)

    for i in tqdm(range(config.num_ssl_iters), desc=f'SSL iters {config.num_ssl_iters}'):

        model = ssl_step(pretrained_model=model, config=config, ssl_iter=i)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SSL config parser")
    parser.add_argument("-pconfig", "--path2config", type=str, 
                        help="The path to yaml config file")
    args = parser.parse_args()

    config = load_config(args.path2config)
    
    ssl_training(config)
