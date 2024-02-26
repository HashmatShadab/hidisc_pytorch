import json
import logging
import os
from functools import partial
from typing import Dict, Any

import dill
import hydra
import numpy as np
import torch
import wandb
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from datasets.loaders import get_dataloaders
from helpers import init_distributed_mode, get_rank, is_main_process, get_world_size, setup_seed
from losses.hidisc import HiDiscLoss
from models import MLP, resnet_backbone, ContrastiveLearningNetwork
from models.resnet_multi_bn_stl import resnet50 as resnet50_multi_bn
from models import resnetv2_50, resnetv2_50_gn
from scheduler import make_optimizer_and_schedule
from train import train_one_epoch
from ft_validate import validate_clean
from utils import save_checkpoints, restart_from_checkpoint
from timm.layers import convert_sync_batchnorm


log = logging.getLogger(__name__)

class HiDiscModel(torch.nn.Module):
    """
    HiDiscModel

    This class represents a High-dimensional Contrastive Learning Model. It is a subclass of `torch.nn.Module`.

    Attributes:
        cf_ (Dict[str, Any]): A dictionary containing the configuration parameters for the model.
        model (ContrastiveLearningNetwork): The main model used for high-dimensional contrastive learning.

    Methods:
        __init__(self, cf: Dict[str, Any]): Initializes the HiDiscModel object.
        forward(self, img): Performs forward pass on the input image.

    """
    def __init__(self, cf: Dict[str, Any]):
        super().__init__()
        self.cf_ = cf

        if cf["model"]["backbone"] == "resnet50":
            bb = partial(resnet_backbone, arch=cf["model"]["backbone"])
        elif cf["model"]["backbone"] == "resnet50_multi_bn":
            bb = partial(resnet50_multi_bn, bn_names=["normal", "pgd"])
        elif cf["model"]["backbone"] == "resnetv2_50":
            bb = partial(resnetv2_50)
        elif cf["model"]["backbone"] == "resnetv2_50_gn":
            bb = partial(resnetv2_50_gn)
        else:
            raise NotImplementedError()

        mlp = partial(MLP,
                      n_in=2048,
                      hidden_layers=cf["model"]["mlp_hidden"],
                      n_out=cf["model"]["num_embedding_out"])
        self.model = ContrastiveLearningNetwork(bb, mlp)



    def forward(self, img, bn_name=None):

        pred = self.model(img, bn_name)

        return pred

    def get_features(self, img, bn_name=None):

        if bn_name is not None:
            out = self.model.bb(img, bn_name)
        else:
            out = self.model.bb(img)
        return out




@hydra.main(version_base=None, config_path="conf", config_name="main")
def main(args):

    """
    Entry point of the program.
    """

    log.info("Info level message")
    # log.debug("Debug level message")


    log.info(f"Current working directory : {os.getcwd()}")
    log.info(f"Orig working directory    : {get_original_cwd()}")

    setup_seed(args['infra']['seed'])

    # Initialize DDP if needed
    init_distributed_mode(args.distributed)

    if is_main_process():
        log.info(OmegaConf.to_yaml(args)) # log.info all the command line arguments

    # Create the output director if not exits
    if get_rank() == 0 and args.out_dir is not None:

        if args.wandb.use:
            wandb.init(project=args.wandb.project, entity=args.wandb.entity, mode=args.wandb.mode,
                       name=args.wandb.exp_name)


        log.info(f"Saving output to {os.path.join(os.getcwd())}")

    train_loader, validation_loader = get_dataloaders(args)

    model = HiDiscModel(args)
    dual_bn = True if args.model.backbone == "resnet50_multi_bn" else False
    model.to(device="cuda")


    def get_n_params(model):
        total = 0
        for p in list(model.parameters()):
            total += np.prod(p.size())
        return total
    log.info(f'==> [Number of parameters of the model is {get_n_params(model)}]')

    update_params = None
    parma_list = model.parameters() if update_params is None else update_params

    # Define loss, create optimizer and scheduler
    num_it_per_ep = len(train_loader)
    log.info(f"==> [Number of iterations per epoch: {num_it_per_ep}], Length of train_loader: {len(train_loader)}, world_size: {get_world_size()}]")
    optimizer, scheduler = make_optimizer_and_schedule(args, model, parma_list, num_it_per_ep)

    crit_params = args["training"]["objective"]["params"]
    criterion = HiDiscLoss(
        lambda_patient=crit_params["lambda_patient"],
        lambda_slide=crit_params["lambda_slide"],
        lambda_patch=crit_params["lambda_patch"],
        supcon_loss_params=crit_params["supcon_params"])

    if args.model.backbone == "resnetv2_50":
        model = convert_sync_batchnorm(model)

    if args.distributed.distributed:

        unused_params = True if args.model.backbone == "resnet50_multi_bn" else False
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.distributed.gpu],
                                                          find_unused_parameters=unused_params,
                                                          broadcast_buffers=False)

    start_epoch, loss = restart_from_checkpoint("checkpoint.pth", model, optimizer, scheduler)

    # Training loop
    strength = 1.0
    for epoch in range(start_epoch, args.training.num_epochs):
        # Train for one epoch
        if args['data']['dynamic_aug']:
            K = 50
            before_strength = strength
            strength = 1.0 - int((epoch/K)) * K / args.training.num_epochs
            if before_strength != strength:
                train_loader, _ = get_dataloaders(args, strength=strength, dynamic_aug=True)
                log.info(f"==> [Dynamic Augmentation: Strength changed from {before_strength} to {strength}]")

        train_stats = train_one_epoch(epoch=epoch, train_loader=train_loader, model=model,
                                      optimizer=optimizer, criterion=criterion, scheduler=scheduler,
                                      attack_type=args.training.attack.name, attack_eps=args.training.attack.eps,
                                      attack_alpha=args.training.attack.alpha,
                                      attack_iters=args.training.attack.iters,
                                      dual_bn=dual_bn,
                                      dynamic_aug=args['data']['dynamic_aug'],
                                      dynamic_strength=strength,
                                      dynamic_weights_lamda=args['training']['dynamic_weights_lamda'],
                                      only_adv = args['training']['only_adv'],
                                      adv_loss_type = args['training']['attack']['loss_type'],
                                      )

        #  Save the checkpoints
        if (epoch + 1) % args.training.save_checkpoint_interval == 0 and is_main_process():
            save_checkpoints(epoch+1, model, optimizer, scheduler, train_stats,
                               name=f'checkpoint_{epoch+1}.pth')

        if is_main_process():
            save_checkpoints(epoch+1, model, optimizer, scheduler, train_stats,
                               name=f'checkpoint.pth')

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
       log.info("Training completed successfully")

if __name__ == "__main__":
    main()
