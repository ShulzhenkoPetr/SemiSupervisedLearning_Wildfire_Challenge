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
from models.models import CustomResNet, NaiveNet, PretrainedResNet
from dataset_utils import get_train_transforms, get_val_transforms, get_dataloader


@hydra.main(version_base=None, config_path="./configs/", config_name="supervised_pretrained")
def train_model(config: SimpleNamespace) -> None:

    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get dataloaders

    train_transforms = get_train_transforms(config, dataset_stats=config.dataset_stats)
    val_transforms = get_val_transforms(dataset_stats=config.dataset_stats)

    train_dataloader = get_dataloader(
        folder_path=Path(config.train_data_path), transforms=train_transforms, batch_size=config.batch_size
    )
    val_dataloader = get_dataloader(
        folder_path=Path(config.val_data_path), transforms=val_transforms, batch_size=config.batch_size, shuffle=False
    )
    # test_dataloader = get_dataloader(
    #     folder_path=config.test_data_path, transforms=val_transforms, batch_size=config.batch_size, shuffle=False
    # )

    # Setup model
    if config.model_name == 'CustomResNet':
        model = CustomResNet(config)
        model.init_weights()
    elif config.model_name == 'NaiveNet':
        model = NaiveNet()
    elif config.model_name == 'PretrainedResNet':
        model = PretrainedResNet(config)
    else:
        raise NotImplementedError(f'This model {config.model.name} is not implemented')
    
    if config.pretrained_checkpoint: 
        checkpoint_dict = torch.load(config.pretrained_checkpoint)
        model.load_state_dict(checkpoint_dict['model_state_dict'])
    
    model.to(device)

    # Setup training

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    if config.optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(
            params=model.parameters(), 
            lr=float(config.lr),
            betas=(0.9, 0.999),
            eps=1e-5,
            weight_decay=float(config.weight_decay)
        )
    else:
        optimizer = torch.optim.SGD(
            params=model.parameters(), 
            lr=float(config.lr),
            weight_decay=float(config.weight_decay),
            momentum=config.momentum
        )
    if config.pretrained_checkpoint and config.load_optimizer:
        optimizer.load_state_dict(checkpoint_dict['optimizer_state_dict'])
    
    if config.sch_step:
        step_scheduler = StepLR(optimizer, step_size=config.sch_step, gamma=config.sch_gamma)
    else:
        step_scheduler = None
    
    def warmup_lr(batch_i, nbr_warmup_batches=300):
        if batch_i < nbr_warmup_batches:
            return (batch_i + 1) / nbr_warmup_batches
        return 1.0
    if config.nbr_warmup_batches:
        warmup_scheduler = LambdaLR(optimizer, lr_lambda=partial(warmup_lr, nbr_warmup_batches=config.nbr_warmup_batches))
    else:
        warmup_scheduler = None

    saving_path = Path(config.saving_path)
    saving_path = saving_path / config.exp_name
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
        name=config.exp_name        
    )

    # Launch training
    for epoch in tqdm(range(config.num_epochs), desc=f'Train {config.num_epochs} epochs'):
        train_epoch(model, 
                    train_dataloader=train_dataloader, 
                    criterion=criterion, 
                    optimizer=optimizer,  
                    logger=logger, config=config, 
                    epoch=epoch, device=device,
                    warmup_scheduler=warmup_scheduler,)
        if step_scheduler:
            step_scheduler.step()
        if val_dataloader is not None:
            val_metrics_dict = evaluate(model, 
                                        dataloader=val_dataloader, 
                                        config=config, 
                                        logger=logger, 
                                        epoch=epoch, 
                                        device=device,
                                        phase_type='val',)

        model_checkpointer.update(val_metrics_dict, model, optimizer, epoch)
        
    model_checkpointer.save()


def train_epoch(model: nn.Module, 
                train_dataloader: torch.utils.data.DataLoader, 
                criterion: nn.Module, 
                optimizer: torch.optim.Optimizer,  
                logger: Any, config: SimpleNamespace, 
                epoch: int, device: torch.device,
                warmup_scheduler: torch.optim.lr_scheduler = None,) -> None:
    
    model.train()
    loss_meter = AverageMeter(name='train/avg_loss', logger=logger)

    for imgs, labels in train_dataloader:
        imgs = imgs.to(device)
        labels = labels.unsqueeze(1).float().to(device)

        optimizer.zero_grad()

        preds = model(imgs)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()
        if warmup_scheduler:
            warmup_scheduler.step()

        loss_meter.update(loss, imgs.shape[0])
    
    if logger:
        loss_meter.log_average(epoch)


def evaluate(
        model: nn.Module, 
        dataloader: torch.utils.data.DataLoader,
        config: SimpleNamespace,
        logger: Any, epoch: int, device: torch.device,
        phase_type: str = 'val',
        is_from_logits: bool = True) -> Dict:
    
    model.eval()

    metrics2meters_dict = {
        name: (get_metric_fn(name, is_from_logits), AverageMeter(name=f'{phase_type}/{name}', logger=logger))
        for name in config.metric_names if get_metric_fn(name, is_from_logits) is not None
    }

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            labels = labels.unsqueeze(1).float().to(device)

            preds = model(imgs)
            for name, (m_fn, m_meter) in metrics2meters_dict.items():
                if 'CE' in name:
                    m_meter.update(m_fn(preds, labels), n=preds.shape[0])
                elif 'Acc' in name:
                    m_meter.update(m_fn(preds, labels, th=config.positive_cls_thr), n=preds.shape[0])
                else:
                    m_meter.update(m_fn(preds, labels, th=config.positive_cls_thr), n=1)

    if logger:
        for name, (m_fn, m_meter) in metrics2meters_dict.items():
            if 'avg' in name:
                m_meter.log_average(epoch)
            else:
                m_meter.log_sum(epoch)

    if 'tp' in config.metric_names:
        # Calculate precision, recall, f1-score
        tp, fp, fn = metrics2meters_dict['tp'][1].sum, metrics2meters_dict['fp'][1].sum, metrics2meters_dict['fn'][1].sum
        precision = precision_fn(tp, fp)
        recall = recall_fn(tp, fn)
        f1_score = f1_score_fn(precision, recall)
        additional_metrics = {
            f'{phase_type}/precision': precision, 
            f'{phase_type}/recall': recall, 
            f'{phase_type}/f1_score': f1_score
        }
    if logger:
        logger.log(additional_metrics)


    res_metrics_dict = {m_meter.name: m_meter.avg if 'avg' in name else m_meter.sum 
                        for name, (_, m_meter) in metrics2meters_dict.items()}
    if 'tp' in metrics2meters_dict.keys():
        res_metrics_dict.update(additional_metrics)

    return res_metrics_dict


    
if __name__ == '__main__':

    # Uncomment to launch without Hydra

    # parser = argparse.ArgumentParser(description="Experiment config parser")
    # parser.add_argument("-pc", "--path2config", type=str, 
    #                     help="The path to yaml config file")
    # args = parser.parse_args()

    # config = load_config(args.path2config)

    # train_model(config)
    train_model()