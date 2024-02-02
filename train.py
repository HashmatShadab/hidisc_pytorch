import torch
from helpers import  accuracy, MetricLogger, plot_grid
import logging

log = logging.getLogger(__name__)



def train_one_epoch(epoch, train_loader, model,
                    optimizer, criterion, print_freq=50):
    # Distributed metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = f'Train: Epoch {epoch}'

    # Switch to training mode
    model.train()

    # Train for one epoch and return the loss and accuracy

    for i, batch in enumerate(metric_logger.log_every(train_loader, print_freq, header)):

        # Move the tensors to the GPUs
        im_reshaped = batch["image"].reshape(-1, *batch["image"].shape[-3:])
        im_reshaped = im_reshaped.to("cuda", non_blocking=True)
        targets = batch["label"].to("cuda", non_blocking=True)
        targets = targets.reshape(-1, 1)


        # Optimize the network for the generated adversarial images
        outputs = model(im_reshaped)
        outputs = outputs.reshape(*batch["image"].shape[:4], outputs.shape[-1])

        # Calculate loss and backpropagate
        losses = criterion(outputs, targets)
        loss = losses["sum_loss"]

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Synchronize
        torch.cuda.synchronize()

        # Measure accuracy
        #acc5 = accuracy(outputs, targets, topk=(5,))[0]

        # Update the losses & top1 accuracy list
        batch_size = batch["image"][0].shape[0]
        metric_logger.update(loss=loss.item()) # other losses
        metric_logger.update(patient_loss=losses["patient_loss"].item()) # patient loss
        # slide loss
        metric_logger.update(slide_loss=losses["slide_loss"].item())
        # patch loss
        metric_logger.update(patch_loss=losses["patch_loss"].item())



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
    print("Averaged stats:", metric_logger)

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
