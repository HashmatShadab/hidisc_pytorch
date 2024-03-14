# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import argparse
import json
import logging
import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from timm.utils import accuracy, AverageMeter
from torch.utils.data import DataLoader

from models.vmamba import VSSM


class ImageNet5k(torchvision.datasets.ImageFolder):

    def __init__(self, image_list="./image_list.json", *args, **kwargs):
        self.image_list = set(json.load(open(image_list, "r"))["images"])
        super(ImageNet5k, self).__init__(is_valid_file=self.is_valid_file, *args, **kwargs)

    def is_valid_file(self, x: str) -> bool:

        file_path = x
        # get image name
        image_name = os.path.basename(file_path)
        # get parent folder name
        folder_name = os.path.basename(os.path.dirname(file_path))

        return f"{folder_name}/{image_name}" in self.image_list



class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        if torch.max(input) > 1:
            input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean.to(device=input.device)) / std.to(
            device=input.device)


def build_vssm_model(model_type, pretrained=False):
    if model_type == "vssm_tiny_0220":
        model = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 4, 2],
            dims=96,
            # ===================
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            # forward_type="v2noz",
            # ===================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # ===================
            drop_path_rate=0.2,
            patch_norm=True,
            norm_layer="ln",
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=False,
        )

    elif model_type == "vssm_small":
        model = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 15, 2],
            dims=96,
            # ===================
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v3noz",
            # ===================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # ===================
            drop_path_rate=0.3 ,
            patch_norm=True,
            norm_layer="ln",
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=False,
        )

    elif model_type == "vssm_base":
        model = VSSM(
            patch_size=4,
            in_chans=3,
            num_classes=1000,
            depths=[2, 2, 15, 2],
            dims=128,
            # ===================
            ssm_d_state=1,
            ssm_ratio=2.0,
            ssm_rank_ratio=2.0,
            ssm_dt_rank="auto",
            ssm_act_layer="silu",
            ssm_conv=3,
            ssm_conv_bias=False,
            ssm_drop_rate=0.0,
            ssm_init="v0",
            forward_type="v3noz",
            # ===================
            mlp_ratio=4.0,
            mlp_act_layer="gelu",
            mlp_drop_rate=0.0,
            # ===================
            drop_path_rate=0.6,
            patch_norm=True,
            norm_layer="ln",
            downsample_version="v3",
            patchembed_version="v2",
            use_checkpoint=False,
        )

    else:
        return None

    return model


def get_model(model_name=None, device="cuda"):
    # load pre-trained models


    if model_name == 'vssm_tiny':

        model = build_vssm_model('vssm_tiny_0220')
        ckpt = torch.load('vssmtiny_dp02_ckpt_epoch_258.pth')
        msg = model.load_state_dict(ckpt['model'], strict=False)
        print(msg)
    elif model_name == 'vssm_small':

        model = build_vssm_model('vssm_small')
        ckpt = torch.load('vmamba_small_e238_ema.pth')
        msg = model.load_state_dict(ckpt['model'], strict=False)
        print(msg)
    elif model_name == 'vssm_base':

        model = build_vssm_model('vssm_base')
        ckpt = torch.load('vssmbase_dp06_ckpt_epoch_241.pth')
        msg = model.load_state_dict(ckpt['model'], strict=False)
        print(msg)

    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

    model = nn.Sequential(Normalize(mean, std), model)

    return model.eval().to(device), mean, std


def plot_grid(w, save=False, name="grid.png"):
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    if save:
        plt.savefig(name)
    plt.show()


@torch.no_grad()
def validate(val_loader, model, logger):
    model.eval()  # set the model to evaluation mode

    batch_time = AverageMeter()
    acc1_meter = AverageMeter()
    acc5_meter = AverageMeter()
    end = time.time()

    for idx, (images, target) in enumerate(val_loader):
        images = images.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)  # Move images and labels to GPU

        # with torch.cuda.amp.autocast(enabled=True):
        outputs = model(images)
        acc1, acc5 = accuracy(outputs, target, topk=(1, 5))
        acc1_meter.update(acc1.item(), target.size(0))
        acc5_meter.update(acc5.item(), target.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if idx % 10 == 0:
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            logger.info(
                f'Test: [{idx}/{len(val_loader)}]\t'
                f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                f'Acc@1 {acc1_meter.val:.3f} ({acc1_meter.avg:.3f})\t'
                f'Acc@5 {acc5_meter.val:.3f} ({acc5_meter.avg:.3f})\t'
                f'Mem {memory_used:.0f}MB')
    logger.info(f' * Acc@1 {acc1_meter.avg:.3f} Acc@5 {acc5_meter.avg:.3f}')

    return acc1_meter.avg


def get_args():
    parser = argparse.ArgumentParser(description='Transferability test')
    parser.add_argument('--dataset', default='imagenet', type=str, help='dataset name', choices=['imagenet', 'imagenet-e'])
    parser.add_argument('--data_dir', help='path to ImageNet dataset', default=r'F:\Code\Projects\MambaInVision\classification\AdvExamples\resnet18_fgsm_eps_2_steps_1\images_labels.pt')
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--source_model_name', default='vssm_tiny')

    args = parser.parse_args()

    return args



if __name__ == "__main__":
    args = get_args()


    data_dir = args.data_dir
    # get parent directory
    parent_dir = os.path.dirname(data_dir)
    # save log path
    if args.dataset == 'imagenet':
        log_dir = os.path.join(parent_dir, f"{args.source_model_name}_eval.log")
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset}")


    logging.basicConfig(filename=log_dir, filemode="a",
                        format="%(name)s → %(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # add console handler to logger
    logger.addHandler(ch)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the model

    model, _,_ = get_model(args.source_model_name, device)

    ine_transform = transforms.Compose([transforms.Resize(256),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        ])

    # Load the dataset
    if args.dataset == 'imagenet':


        # dataset = ImageNet5k(root=args.data_dir, transform=ine_transform)
        dataset = torchvision.datasets.ImageFolder(root=args.data_dir, transform=ine_transform)


        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        acc = validate(dataloader, model, logger)
        logger.info(f"Accuracy: {acc}")




