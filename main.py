import argparse
import json
import os

import logging
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf


import numpy as np
import torch
import torch.nn as nn
import dill
import wandb

from helpers import init_distributed_mode, get_rank, is_main_process, get_mean_std, get_world_size
from scheduler import make_optimizer_and_schedule
from train import train_one_epoch
from validate import validate_clean

from models import MLP, resnet_backbone, ContrastiveLearningNetwork
from models.resnet_multi_bn import resnet50 as resnet50_multi_bn
from losses.hidisc import HiDiscLoss
from common import (setup_output_dirs, parse_args, get_exp_name,
                                 config_loggers, get_optimizer_func,
                                 get_scheduler_func, get_dataloaders)


log = logging.getLogger(__name__)

"""
ImageNet Training:
Epochs: 30
Batch Size: 256
Optimizer: SGD
Learning Rate: 0.001
Momentum: 0.9
Weight Decay: 0.0001
Column ablation of fixed width = 19
Augmnetation: RandomResizedCrop(224), RandomHorizontalFlip(), ColorJitter()

CIFAR-10 Training:
Epochs: 30
Batch Size: 128
Optimizer: SGD
Learning Rate: 10E-2
Scheduler: StepLR(10, 0.1)
Momentum: 0.9
Weight Decay: 5 x 10E-4
Column ablation of fixed width = 4
Augmnetation: RandomHorizontalFlip()
Column ablation and then upsampling to 224x224

"""
from functools import partial
from typing import Dict, Any
import torchmetrics


class HiDiscModel(torch.nn.Module):

    def __init__(self, cf: Dict[str, Any]):
        super().__init__()
        self.cf_ = cf

        if cf["model"]["backbone"] == "resnet50":
            bb = partial(resnet_backbone, arch=cf["model"]["backbone"])
        elif cf["model"]["backbone"] == "resnet50_multi_bn":
            bb = partial(resnet50_multi_bn)
        else:
            raise NotImplementedError()

        mlp = partial(MLP,
                      n_in=bb().num_out,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
        self.model = ContrastiveLearningNetwork(bb, mlp)



    def forward(self, img):

        pred = self.model(img)
        return pred




@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(args):
    log.info("Info level message")
    # log.debug("Debug level message")


    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {get_original_cwd()}")

    # Initialize DDP if needed
    init_distributed_mode(args.distributed)

    if is_main_process():
        log.info(OmegaConf.to_yaml(args)) # log.info all the command line arguments

    # Create the output director if not exits
    if get_rank() == 0 and args.out_dir is not None:

        if args.wandb.use:
            name = args.out_dir.replace("/", "_")
            wandb.init(project=args.wandb.project, entity=args.wandb.entity, mode=args.wandb.mode,
                       name=name, group=args.wandb.exp_name)
            config = wandb.config
            config.update(args)

        log.info(f"Saving output to {os.path.join(os.getcwd())}")

    train_loader, validation_loader = get_dataloaders(args)

    model = HiDiscModel(args)
    model.to(device="cuda")


    def get_n_params(model):
        total = 0
        for p in list(model.parameters()):
            total += np.prod(p.size())
        return total
    log.info(f'==> [Number of parameters of the model is {get_n_params(model)}]')

    start_epoch = 0
    if  args.eval_only and not args.model.restart_from_ckpt:
        """
        Loads only model weights for evaluation.
        if restart_from_ckpt, then model weights will be loaded from the output dir,
        no need to pass the checkpoints path.
         
        """
        checkpoint = torch.load(args.model.checkpoints_path, pickle_module=dill)
        start_epoch = checkpoint["epoch"]
        log.info(f"Evaluating Model from {args.model.checkpoints_path}, Epoch : {start_epoch}")
        model_weights = checkpoint["model"]
        # remove the module from the keys
        model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}
        model_weights = {k.replace("model.model", "model"): v for k, v in model_weights.items()}
        msg = model.load_state_dict(model_weights, strict=False)
        log.info("Load model with msg: ", msg)



    """
    
    Freeze the model layers if needed. It only works for resnet models.
    By default it is -1 and returns None, which means no layers are frozen.
    
    """
    update_params = None
    parma_list = model.parameters() if update_params is None else update_params

    # Define loss, create optimizer and scheduler
    num_it_per_ep = len(train_loader) // get_world_size()
    optimizer, scheduler = make_optimizer_and_schedule(args, model, parma_list, num_it_per_ep)

    crit_params = args["training"]["objective"]["params"]
    criterion = HiDiscLoss(
        lambda_patient=crit_params["lambda_patient"],
        lambda_slide=crit_params["lambda_slide"],
        lambda_patch=crit_params["lambda_patch"],
        supcon_loss_params=crit_params["supcon_params"])


    if args.distributed.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.distributed.gpu],
                                                          find_unused_parameters=args.distributed.find_unused_params)

    # start_epoch, best_smooth_acc, best_certified_acc, best_clean_std_acc = restart_from_checkpoint(args.out_dir,
    #                                                                                                "checkpoint.pth",
    #                                                                                                model, optimizer,
    #                                                                                                scheduler)
    # Perform evaluation and exit if `eval_only` is set
    if args.eval_only:
        # Validate the model
        validate_clean(validation_loader, model, criterion)
        return  # Safe exit


    # Training loop
    for epoch in range(start_epoch, args.training.num_epochs):
        # Train for one epoch
        train_stats = train_one_epoch(epoch=epoch, train_loader=train_loader, model=model,
                                      optimizer=optimizer, criterion=criterion, scheduler=scheduler)
        # if scheduler:
        #     scheduler.step()

        #  Save the checkpoints
        # save_checkpoints(epoch + 1, model, optimizer, scheduler, train_stats, train_stats,
        #                   name='checkpoint.pth', outdir=args.out_dir)





        # Log the epoch stats
        log_stats_train = {
            'Epoch': epoch,
            **{f'train_{key}': value for key, value in train_stats.items() if "acc5" not in key},
        }

        if args.out_dir and is_main_process():
            with open("log.txt", mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats_train) + "\n")
            if args.wandb.use:
                wandb.log(log_stats_train)


    if is_main_process():
       log.info("Hiiii")

if __name__ == "__main__":
    best_certified_accuracy = main()
