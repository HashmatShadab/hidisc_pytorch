import torch
import os
import gdown
import torch.nn as nn
from timm.models import create_model
from ares.utils.model import NormalizeByChannelMeanStd
from ares.model.resnet import resnet50, wide_resnet50_2, ResNetGELU
from ares.model.resnet_denoise import resnet152_fd
from ares.model import vit_mae
from ares.model.imagenet_model_zoo import imagenet_model_zoo
from ares.utils.registry import registry
import torchvision
from torchvision.transforms.functional import normalize

@registry.register_model('ImageNetCLS')
class ImageNetCLS(torch.nn.Module):
    '''The class to create ImageNet model.'''
    def __init__(self, model_name, normalize=True):
        '''
        Args:
            model_name (str): The model name in the ImageNet model zoo.
            normalize (bool): Whether interating the normalization layer into the model.
        '''
        super(ImageNetCLS).__init__()
        self.model_name = model_name
        self.normalize = normalize
        self.backbone = imagenet_model_zoo[self.model_name]['model']
        mean=imagenet_model_zoo[self.model_name]['mean']
        std=imagenet_model_zoo[self.model_name]['std']
        self.pretrained=imagenet_model_zoo[self.model_name]['pretrained']
        act_gelu=imagenet_model_zoo[self.model_name]['act_gelu']
    
        if self.backbone=='resnet50_rl':
            model=resnet50()
        elif self.backbone=='wide_resnet50_2_rl':
            model=wide_resnet50_2()
        elif self.backbone=='resnet152_fd':
            model = resnet152_fd()
        elif self.backbone=='vit_base_patch16' or self.backbone=='vit_large_patch16':
            model=vit_mae.__dict__[self.backbone](num_classes=1000, global_pool='')
        else:
            model_kwargs=dict({'num_classes': 1000})
            if act_gelu:
                model_kwargs['act_layer']=ResNetGELU
            model = create_model(self.backbone, pretrained=self.pretrained, **model_kwargs)
        self.model=model

        self.url = imagenet_model_zoo[self.model_name]['url']

        ckpt_name = '' if self.pretrained else imagenet_model_zoo[self.model_name]['pt']
        self.model_path=os.path.join(registry.get_path('cache_dir'), ckpt_name)
        
        if self.url:
            gdown.download(self.url, self.model_path, quiet=False, resume=True)

        self.load()
        
        if self.normalize:
            normalization = NormalizeByChannelMeanStd(mean=mean, std=std)
            self.model = torch.nn.Sequential(normalization, self.model)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): The input images. The images should be torch.Tensor with shape [N, C, H, W] and range [0, 1].

        Returns:
            torch.Tensor: The output logits with shape [N D].

        '''

        labels = self.model(x)
        return labels

    def load(self):
        '''The function to load ckpt.'''
        if not self.pretrained:
            ckpt=torch.load(self.model_path, map_location='cpu')
            self.model.load_state_dict(ckpt)


@registry.register_model('RobustImageNetEncoders')
class RobustImageNetEncoders(torch.nn.Module):
    '''The class to create ImageNet model.'''

    def __init__(self, model_name, normalize=True):
        '''
        Args:
            model_name (str): The model name in the ImageNet model zoo.
            normalize (bool): Whether interating the normalization layer into the model.
        '''
        super(RobustImageNetEncoders, self).__init__()
        self.model_name = model_name
        self.normalize = normalize
        self.backbone = imagenet_model_zoo[self.model_name]['model']
        self.mean = imagenet_model_zoo[self.model_name]['mean']
        self.std = imagenet_model_zoo[self.model_name]['std']
        self.pretrained = imagenet_model_zoo[self.model_name]['pretrained']
        act_gelu = imagenet_model_zoo[self.model_name]['act_gelu']

        if self.backbone == 'resnet50_rl':
            model = resnet50(num_classes=0)
        elif self.backbone == 'wide_resnet50_2_rl':
            model = wide_resnet50_2(num_classes=0)
        elif self.backbone == 'resnet152_fd':
            model = resnet152_fd(num_classes=0)
        elif self.backbone == 'vit_base_patch16' or self.backbone == 'vit_large_patch16':
            model = vit_mae.__dict__[self.backbone](num_classes=0, global_pool='')
        else:
            model_kwargs = dict({'num_classes': 0})
            if act_gelu:
                model_kwargs['act_layer'] = ResNetGELU
            if self.model_name == 'vits_at':
                # add  'img_size': (224, 224)}
                model_kwargs['img_size'] = (224, 224)
                model_kwargs['dynamic_img_size'] = True
            model = create_model(self.backbone, pretrained=self.pretrained, **model_kwargs)
        self.model = model #  'img_size': (224, 224)}

        self.url = imagenet_model_zoo[self.model_name]['url']

        ckpt_name = '' if self.pretrained else imagenet_model_zoo[self.model_name]['pt']
        self.model_path = os.path.join(registry.get_path('cache_dir'), ckpt_name)

        if self.url:
            gdown.download(self.url, self.model_path, quiet=False, resume=True)

        self.load()
        # self.get_normalizer = torchvision.transforms.Normalize(mean=self.mean, std=self.std)
        # if self.normalize:
        #     normalization = NormalizeByChannelMeanStd(mean=mean, std=std)
        #     self.model = torch.nn.Sequential(normalization, self.model)

    def forward(self, x):
        '''
        Args:
            x (torch.Tensor): The input images. The images should be torch.Tensor with shape [N, C, H, W] and range [0, 1].

        Returns:
            torch.Tensor: The output logits with shape [N D].

        '''
        if self.normalize:
            logits = self.model(normalize(x, mean=self.mean, std=self.std))
        else:
            logits = self.model(x)
        return logits

    def forward_features(self, x):
        '''
        Args:
            x (torch.Tensor): The input images. The images should be torch.Tensor with shape [N, C, H, W] and range [0, 1].

        Returns:
            torch.Tensor: The output features with shape [N D].

        '''

        features = self.model(x)
        return features

    # def get_normalizer(self):
    #     return torchvision.transforms.Normalize(mean=self.mean, std=self.std)

    def load(self):
        '''The function to load ckpt.'''
        if not self.pretrained:
            ckpt = torch.load(self.model_path, map_location='cpu')
            msg = self.model.load_state_dict(ckpt, strict=False)
            print(msg)
