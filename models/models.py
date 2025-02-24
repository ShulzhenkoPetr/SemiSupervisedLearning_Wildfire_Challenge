from typing import Dict, List, Tuple
from types import SimpleNamespace
from pathlib import Path
import torch
from torch import nn
from torchvision import models, transforms

from .blocks import ResBlock, BasicBlock


def load_checkpoint(path2ckpt: Path, config: SimpleNamespace):
    checkpoint_dict = torch.load(path2ckpt)
    if config.model_name == 'CustomResNet':
        model = CustomResNet(config)
    elif config.model_name == 'NaiveNet':
        model = NaiveNet()
    elif config.model_name == 'PretrainedResNet':
        model = PretrainedResNet(config)
    else:
        raise NotImplementedError(f'This model {config.model_name} is not implemented')
    
    model.load_state_dict(checkpoint_dict['model_state_dict'])

    return model


class CustomResNet(nn.Module):

    def __init__(self, model_config: SimpleNamespace):
        super().__init__()

        architecture_spec_list = model_config.architecture_list

        self.layers = nn.ModuleList()

        for i, spec_tuple in enumerate(architecture_spec_list):
            if i == 0:
                self.layers.append(
                    BasicBlock(
                        in_feats=int(spec_tuple[1]),
                        out_feats=int(spec_tuple[2]),
                        kernel_size=7, stride=2, padding=3
                    )
                )
                self.layers.append(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
            else:
                self.layers.append(
                    ResBlock(
                        in_feats=spec_tuple[1],
                        out_feats=spec_tuple[2],
                        kernel_size=3, stride=1, padding=1,
                        downsample=None if spec_tuple[0] == 'conv' else nn.MaxPool2d(kernel_size=2, stride=2)
                    )
                )
        self.layers.append(nn.AdaptiveAvgPool2d((1, 1)))
        self.layers.append(nn.Flatten())
        self.layers.append(nn.Linear(in_features=architecture_spec_list[-1][-1], out_features=model_config.nbr_classes))
    

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
    
    def init_weights(self, ):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


class NaiveNet(nn.Module):
    def __init__(self, num_classes: int = 1):
        super().__init__()

        self.features = nn.Sequential(
            BasicBlock(3, 16),  
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicBlock(16, 32),  
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicBlock(32, 64),  
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicBlock(64, 128),  
            nn.MaxPool2d(kernel_size=2, stride=2),

            BasicBlock(128, 256),  
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  
            nn.Flatten(),  
            nn.Linear(256, num_classes)  
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
    

class PretrainedResNet(nn.Module):
    '''Adapts a pretrained ResNet to the task and freezes selected number of layers'''
    def __init__(self, config: SimpleNamespace):
        super().__init__()
        
        if hasattr(models, config.pretrained_resnet):
            self.resnet = getattr(models, config.pretrained_resnet)(pretrained=True)
        else:
            raise NotImplementedError(f'No such pretrained ResNet {config.pretrained_resnet}')
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, config.nbr_classes)
        
        if config.freeze_layers:
            for name, param in self.resnet.named_parameters():
                if any(layer in name for layer in config.unfreeze_layers_list):
                    param.requires_grad = True
                else:
                    param.requires_grad = False

    def forward(self, x):
        return self.resnet(x)
    

class TTAWrapper(nn.Module):
    '''Provides Test-Time Augmentations prediction for a given base model'''
    def __init__(self, base_model: nn.Module, th: float = 0.5):
        super().__init__()

        self.base_model = base_model
        self.th = th
        self.transforms_list = [
            transforms.RandomRotation((90, 90)),
            transforms.RandomRotation((180, 180)),
            transforms.RandomRotation((270, 270))
        ]
        # self.transforms_list = [
        #     transforms.RandomHorizontalFlip(p=1),
        #     transforms.RandomVerticalFlip(p=1)
        # ]

    def forward(self, x, return_label=True):
        pred_scores = [torch.sigmoid(self.base_model(x))]
        for transf in self.transforms_list:
            pred_scores.append(torch.sigmoid(self.base_model(transf(x))))
        
        stacked_scores = torch.cat(pred_scores, dim=1)
        if return_label:
            return (stacked_scores.mean(dim=1, keepdim=True) >= self.th).to(dtype=torch.int)
        else:
            return stacked_scores.mean(dim=1, keepdim=True)


class MajorityVotingEnsemble(nn.Module):

    def __init__(self, model_list: List[Tuple[Path, SimpleNamespace]], device: torch.device):
        super().__init__()

        self.models = nn.ModuleList([load_checkpoint(ckpt_path, config).to(device) 
                                     for ckpt_path, config in model_list])
        self.thresholds = [config.positive_cls_thr for _, config in model_list]
        
        self.eval()

    def forward(self, x):
        votes = [torch.sigmoid(model(x)) >= th for model, th in zip(self.models, self.thresholds)]
        stacked_votes = torch.cat(votes, dim=1)

        num_models = stacked_votes.shape[1]
        threshold = num_models // 2

        # Pushes to FNs
        return stacked_votes.sum(dim=1, keepdim=True) > int((len(votes) / 2))

        # Pushes to FPs
        # if num_models % 2 == 0: 
        #     majority_vote = stacked_votes.sum(dim=1, keepdim=True) >= threshold
        # else:
        #     majority_vote = stacked_votes.sum(dim=1, keepdim=True) > threshold
        # return majority_vote






if __name__ == '__main__':

    model_cnfg = {'nbr_classes': 1}
    input_tensor = torch.randn(1, 3, 224, 224)

    model = CustomResNet(model_cnfg)
    model.init_weights()
    with torch.no_grad():
        res = model(input_tensor)

    print(res.shape)