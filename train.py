import torch
from helpers import  accuracy, MetricLogger, plot_grid
import logging
import torchvision

log = logging.getLogger(__name__)
def plot_grid(w, save=False, name="grid.png"):
    import matplotlib.pyplot as plt
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    if save:
        plt.savefig(name)
    plt.show()


import torch


def pgd_attack(model, criterion, images, targets, shape, eps=1 / 255, alpha=1 / 255, iters=2):
    # Initialize delta and set its bounds
    delta = torch.zeros_like(images).cuda()
    delta.uniform_(-eps, eps)
    delta = torch.clamp(delta, 0 - images, 1 - images)
    delta.requires_grad = True

    # Attack
    for _ in range(iters):
        adv_images = images + delta
        # Forward pass to the model
        outputs = model(adv_images)
        outputs = outputs.reshape(*shape, outputs.shape[-1])

        losses = criterion(outputs, targets)
        loss = losses["sum_loss"]

        # Backpropagate & estimate delta
        loss.backward()

        grad = delta.grad.detach()
        d = delta[:, :, :, :]
        g = grad[:, :, :, :]

        images_ = images[:, :, :, :]
        d = torch.clamp(d + alpha * torch.sign(g), min=-eps, max=eps)
        d = torch.clamp(d, 0 - images_, 1 - images_)
        delta.data[:, :, :, :] = d
        delta.grad.zero_()

    # Create final adversarial images
    adv_images = images + delta

    return adv_images.detach()




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
        targets = targets.reshape(-1, 1)

        # adv_images = pgd_attack(model=model, criterion=criterion, targets=targets, images=im_reshaped, eps=1/255, alpha=1/255, iters=7, shape=batch["image"].shape[:4])
        # adv_outputs = model(adv_images)
        # adv_outputs = adv_outputs.reshape(*batch["image"].shape[:4], adv_outputs.shape[-1])
        # adv_losses = criterion(adv_outputs, targets)
        # adv_loss = adv_losses["sum_loss"]
        # if using Dual Stream model, then use the following line
        # model(inputs, 'normal') for clean images and model(inputs, 'pgd') for adversarial images
        # also fopr attack pass model(images + delta, 'pgd')


        clean_outputs = model(im_reshaped)
        clean_outputs = clean_outputs.reshape(*batch["image"].shape[:4], clean_outputs.shape[-1])

        clean_losses = criterion(clean_outputs, targets)
        clean_loss = clean_losses["sum_loss"]

        optimizer.zero_grad()
        clean_loss.backward()
        optimizer.step()
        scheduler.step()

        # Synchronize
        torch.cuda.synchronize()


        metric_logger.update(loss=clean_loss.item())
        metric_logger.update(patient_loss=clean_losses["patient_loss"].item())
        # slide loss
        metric_logger.update(slide_loss=clean_losses["slide_loss"].item())
        # patch loss
        metric_logger.update(patch_loss=clean_losses["patch_loss"].item())



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
