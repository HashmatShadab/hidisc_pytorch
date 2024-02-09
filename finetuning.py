import json
import logging
import os
from functools import partial
from typing import Dict, Any
import torch
import torchvision

import dill
import hydra
import numpy as np
import torch
import wandb
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

from datasets.loaders import get_dataloaders
from helpers import init_distributed_mode, get_rank, is_main_process, get_world_size
from models import MLP, resnet_backbone, ContrastiveLearningNetwork
from models.resnet_multi_bn import resnet50 as resnet50_multi_bn
from scheduler import make_optimizer_and_schedule
from utils import save_checkpoints, restart_from_checkpoint
import torch
from helpers import  accuracy, MetricLogger, setup_seed
log = logging.getLogger(__name__)


class FineTuneModel(torch.nn.Module):
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

    def __init__(self, cf: Dict[str, Any], num_classes: int):
        super().__init__()
        self.cf_ = cf

        if cf["model"]["backbone"] == "resnet50":
            backbone = resnet_backbone(arch=cf["model"]["backbone"])
        elif cf["model"]["backbone"] == "resnet50_multi_bn":
            backbone = resnet50_multi_bn()
        else:
            raise NotImplementedError()

        self.bb = backbone
        self.linear_head = torch.nn.Linear(in_features=backbone.num_out, out_features=num_classes)


    def forward(self, img):

        features = self.bb(img)
        logits = self.linear_head(features)

        return logits


@hydra.main(version_base=None, config_path="conf", config_name="ft")
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
        log.info(OmegaConf.to_yaml(args))  # log.info all the command line arguments

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

    model = FineTuneModel(args, num_classes=7)
    model.to(device="cuda")

    def get_n_params(model):
        total = 0
        for p in list(model.parameters()):
            total += np.prod(p.size())
        return total

    log.info(f'==> [Number of parameters of the model is {get_n_params(model)}]')

    if args.model.start_from_ssl_ckpt:
        """
        Loads only model weights for evaluation.
        if restart_from_ckpt, then model weights will be loaded from the output dir,
        no need to pass the checkpoints path.

        """
        log.info(f"Loading model weights from {args.model.start_from_ssl_ckpt}")
        checkpoint = torch.load(args.model.start_from_ssl_ckpt, pickle_module=dill)
        model_weights = checkpoint["model"]
        # remove the module from the keys
        model_weights = {k.replace("module.", ""): v for k, v in model_weights.items()}
        # remove model.model from the keys
        model_weights = {k.replace("model.", ""): v for k, v in model_weights.items()}

        msg = model.load_state_dict(model_weights, strict=False)
        log.info(f"Load model with msg: {msg}")



    if args.model.finetuning == 'linear':
        # Freeze the backbone
        for param in model.bb.parameters():
            param.requires_grad = False
    elif args.model.finetuning == 'full':
        for param in model.bb.parameters():
            param.requires_grad = True
    else:
        raise NotImplementedError()

    parma_list = list(filter(lambda p: p.requires_grad, model.parameters()))

    # Define loss, create optimizer and scheduler
    num_it_per_ep = len(train_loader) // get_world_size()
    optimizer, scheduler = make_optimizer_and_schedule(args, model, parma_list, num_it_per_ep)

    criterion = torch.nn.CrossEntropyLoss()

    # print total number of learnable parameters
    if is_main_process():
        log.info(f"Total number of learnable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")

    if args.distributed.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.distributed.gpu],
                                                          find_unused_parameters=args.distributed.find_unused_params)

    start_epoch, loss = restart_from_checkpoint("checkpoint.pth", model, optimizer, scheduler)
    args.training.num_epochs = args.training.num_epochs - start_epoch

    # Perform evaluation and exit if `eval_only` is set
    if args.eval_only:
        # Validate the model
        validate_clean(validation_loader, model, criterion)
        return  # Safe exit

    # Training loop
    for epoch in range(args.training.num_epochs):
        # Train for one epoch
        train_stats = train_one_epoch(epoch=epoch, train_loader=train_loader, model=model,
                                      optimizer=optimizer, criterion=criterion, scheduler=scheduler)

        val_stats = validate_clean(validation_loader, model, criterion)
        # if scheduler:
        #     scheduler.step()

        #  Save the checkpoints
        save_checkpoints(epoch + 1, model, optimizer, scheduler, train_stats,
                         name='checkpoint.pth')

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






def train_one_epoch(epoch, train_loader, model,
                    optimizer, criterion, scheduler, print_freq=50):

    """
    :param epoch: The current epoch number.
    :param train_loader: The data loader for training data.
    :param model: The model to be trained.
    :param optimizer: The optimizer used to update the model's parameters.
    :param criterion: The loss function used to calculate the loss.
    :param scheduler: The learning rate scheduler.
    :param print_freq: The frequency at which to print the training progress.
    :return: A dictionary containing the averaged metrics from the training epoch.

    """

    # Distributed metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Train: Epoch {epoch}'

    # Switch to training mode
    model.train()


    for i, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):

        # Move the tensors to the GPUs
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        orig_im = batch['base_image'].reshape(-1, *batch['base_image'].shape[-3:])
        im_reshaped = im_reshaped.to("cuda", non_blocking=True)
        targets = batch["label"].to("cuda", non_blocking=True)

        # adv_images = pgd_attack(model=model, criterion=criterion, targets=targets, images=im_reshaped, eps=1/255, alpha=1/255, iters=7, shape=batch["image"].shape[:4])
        # adv_outputs = model(adv_images)
        # adv_outputs = adv_outputs.reshape(*batch["image"].shape[:4], adv_outputs.shape[-1])
        # adv_losses = criterion(adv_outputs, targets)
        # adv_loss = adv_losses["sum_loss"]
        # if using Dual Stream model, then use the following line
        # model(inputs, 'normal') for clean images and model(inputs, 'pgd') for adversarial images
        # also fopr attack pass model(images + delta, 'pgd')


        clean_outputs = model(im_reshaped)

        clean_loss = criterion(clean_outputs, targets)

        optimizer.zero_grad()
        clean_loss.backward()
        optimizer.step()
        scheduler.step()

        # Synchronize
        torch.cuda.synchronize()


        metric_logger.update(loss=clean_loss.item())



        # Add LRs to the metric logger ass well
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])
        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info(f"Averaged stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}




def validate_clean(val_loader, model, criterion):
    # Distributed metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # Switch to evaluation mode
    model.eval()

    # Evaluate and return the loss and accuracy
    with torch.no_grad():
        for i, batch in enumerate(metric_logger.log_every(val_loader, 10, header)):
            # Move the tensors to the GPUs
            # Move the tensors to the GPUs
            im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
            im_reshaped = im_reshaped.to("cuda", non_blocking=True)
            targets = batch["label"].to("cuda", non_blocking=True)

            # Forward pass to the network
            outputs = model(im_reshaped)


            # Calculate loss
            loss = criterion(outputs, targets)

            # Measure accuracy
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            acc5 = accuracy(outputs, targets, topk=(5,))[0]

            # Update the losses & top1 accuracy list
            batch_size = im_reshaped.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    log.info(f"* Acc@1 {metric_logger.acc1.global_avg:.3f} Acc@5 {metric_logger.acc5.global_avg:.3f} loss {metric_logger.loss.global_avg:.3f}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    best_certified_accuracy = main()
