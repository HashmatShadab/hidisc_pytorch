import logging

import torch
import torchvision

from helpers import MetricLogger
import torch.nn.functional as F

log = logging.getLogger(__name__)
def plot_grid(w, save=False, name="grid.png"):
    import matplotlib.pyplot as plt
    grid_img = torchvision.utils.make_grid(w)
    plt.imshow(grid_img.permute(1,2,0).cpu())
    if save:
        plt.savefig(name)
    plt.show()




def pgd_attack(model, criterion, images, targets, shape, eps=1 / 255, alpha=1 / 255, iters=2, dual_bn=False):


    log.info(f"PGD Attack Model is in train mode: {model.training}")

    delta = torch.rand_like(images) * eps * 2 - eps
    delta = torch.nn.Parameter(delta)

    # Attack
    for _ in range(iters):
        # Forward pass to the model
        if dual_bn:
            outputs = model(images+delta, 'pgd')
        else:
            outputs = model(images + delta)

        outputs = outputs.reshape(*shape, outputs.shape[-1])

        model.zero_grad()

        losses = criterion(outputs, targets)
        loss = losses["sum_loss"]
        # Backpropagate & estimate delta
        loss.backward()

        delta.data = delta.data + alpha * delta.grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

        log.info(f"PGD Attack Loss {_}: {loss.item()}")

    # Create final adversarial images
    adv_images = images + delta

    return adv_images.detach()


def pgd_attack_2(model, criterion, images, targets, shape, eps=1 / 255, alpha=1 / 255, iters=2, dual_bn=False):


    # Initialize delta and set its bounds
    delta = torch.zeros_like(images).cuda()
    delta.uniform_(-eps, eps)
    delta = torch.clamp(delta, 0 - images, 1 - images)
    delta.requires_grad = True

    log.info(f"PGD Attack Model is in train mode: {model.training}")


    # Attack
    for _ in range(iters):
        adv_images = images + delta
        # Forward pass to the model
        if dual_bn:
            outputs = model(adv_images, 'pgd')
        else:
            outputs = model(adv_images)

        outputs = outputs.reshape(*shape, outputs.shape[-1])

        model.zero_grad()

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
        # log the loss at each iteration
        log.info(f"PGD Attack Loss {_}: {loss.item()}")

    # Create final adversarial images
    adv_images = images + delta

    return adv_images.detach()




def train_one_epoch(epoch, train_loader, model,
                    optimizer, criterion, scheduler, print_freq=50, attack_type='pgd',
                    attack_eps=1/255, attack_alpha=1/255, attack_iters=7, dual_bn=False,
                    dynamic_aug=False, dynamic_weights_lamda=0.5, dynamic_strength=1.0):

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
        orig_im = orig_im.to("cuda", non_blocking=True)
        targets = batch["label"].to("cuda", non_blocking=True)
        targets = targets.reshape(-1, 1)

        if attack_type == 'pgd':
            # put model in eval mode to generate adversarial examples
            model.eval()
            adv_images = pgd_attack(model=model, criterion=criterion, targets=targets, images=im_reshaped, eps=attack_eps/255.0,
                                    alpha=attack_alpha/255.0, iters=attack_iters, shape=batch["image"].shape[:4], dual_bn=dual_bn)
            adv_outputs = model(adv_images, 'pgd') if dual_bn else model(adv_images)
            zi_adv = adv_outputs[[int(2 * i) for i in range(adv_outputs.shape[0] // 2)]]
            zj_adv = adv_outputs[[int(2 * i + 1) for i in range(adv_outputs.shape[0] //2)]]
            adv_outputs = adv_outputs.reshape(*batch["image"].shape[:4], adv_outputs.shape[-1])
            adv_losses = criterion(adv_outputs, targets)
            adv_loss = adv_losses["sum_loss"]
            # put model back in train mode
            # print to check if model is back in train mode
            model.train()
            logging.info(f"Model is in train mode: {model.training}")

        elif attack_type == 'pgd_2':
            # put model in eval mode to generate adversarial examples
            model.eval()
            adv_images = pgd_attack_2(model=model, criterion=criterion, targets=targets, images=im_reshaped, eps=attack_eps/255.0,
                                    alpha=attack_alpha/255.0, iters=attack_iters, shape=batch["image"].shape[:4], dual_bn=dual_bn)
            adv_outputs = model(adv_images, 'pgd') if dual_bn else model(adv_images)
            zi_adv = adv_outputs[[int(2 * i) for i in range(adv_outputs.shape[0] // 2)]]
            zj_adv = adv_outputs[[int(2 * i + 1) for i in range(adv_outputs.shape[0] // 2)]]
            adv_outputs = adv_outputs.reshape(*batch["image"].shape[:4], adv_outputs.shape[-1])
            adv_losses = criterion(adv_outputs, targets)
            adv_loss = adv_losses["sum_loss"]
            # put model back in train mode
            # print to check if model is back in train mode
            model.train()
            logging.info(f"Model is in train mode: {model.training}")
        else:
            adv_loss = 0
            log.info("No attack type specified,  Adv loss set to 0.0")

        z_orig = model(orig_im, 'normal') if dual_bn else model(orig_im)

        clean_outputs = model(im_reshaped, 'normal') if dual_bn else model(im_reshaped)
        zi = clean_outputs[[int(2 * i) for i in range(clean_outputs.shape[0] //2)]]
        zj = clean_outputs[[int(2 * i + 1) for i in range(clean_outputs.shape[0] // 2)]]
        clean_outputs = clean_outputs.reshape(*batch["image"].shape[:4], clean_outputs.shape[-1])

        clean_losses = criterion(clean_outputs, targets)
        clean_loss = clean_losses["sum_loss"]

        # reshape zi, zj, z_orig, zi_adv, zj_adv into (bs, -1)
        zi = zi.reshape(zi.shape[0], -1)
        zj = zj.reshape(zj.shape[0], -1)
        z_orig = z_orig.reshape(z_orig.shape[0], -1)
        zi_adv = zi_adv.reshape(zi_adv.shape[0], -1)
        zj_adv = zj_adv.reshape(zj_adv.shape[0], -1)

        sir_loss, air_loss = reg_loss_(z_orig, zi, zi_adv, zj, zj_adv)

        if dynamic_aug:
            weight = dynamic_weights_lamda*(1 - dynamic_strength)
        else:
            weight = 0.0

        total_loss = (1 - weight)*clean_loss + (1 + weight)*adv_loss + 0.5*sir_loss + 0.5*air_loss


        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        # Synchronize
        torch.cuda.synchronize()

        # Update the metric logger with clean and adversarial losses
        metric_logger.update(total_loss=total_loss.item())
        metric_logger.update(clean_loss=clean_loss.item())

        # Update the metric logger with individual losses from clean loss
        metric_logger.update(clean_patient_loss=clean_losses["patient_loss"].item())
        metric_logger.update(clean_slide_loss=clean_losses["slide_loss"].item())
        metric_logger.update(clean_patch_loss=clean_losses["patch_loss"].item())
        metric_logger.update(sir_loss=sir_loss.item())
        metric_logger.update(air_loss=air_loss.item())

        # Update the metric logger with the weight coefficient
        metric_logger.update(weight_coefficent=weight)

        # Update the metric logger with individual losses from adversarial loss
        if attack_type == 'pgd':
            metric_logger.update(adv_loss=adv_loss.item())
            metric_logger.update(adv_patient_loss=adv_losses["patient_loss"].item())
            metric_logger.update(adv_slide_loss=adv_losses["slide_loss"].item())
            metric_logger.update(adv_patch_loss=adv_losses["patch_loss"].item())
            metric_logger.update(avg_increase=adv_loss.item() - clean_loss.item())

        else:
            metric_logger.update(adv_loss=0.0)
            metric_logger.update(adv_patient_loss=0.0)
            metric_logger.update(adv_slide_loss=0.0)
            metric_logger.update(adv_patch_loss=0.0)
            metric_logger.update(avg_increase=0.0)


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
    # log.info(f"Average stats: {metric_logger}")

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def reg_loss(zo_norm, zi_norm, zi_adv_norm, zj_norm, zj_adv_norm, temp=0.5):


    bs = zi_norm.shape[0]
    mask = torch.ones((bs, bs), dtype=bool).fill_diagonal_(0)

    ### Standard Invariant Regularization (Implementation follows https://github.com/NightShade99/Self-Supervised-Vision/blob/42183d391f51c60383ff0cb044e2d71379aa7461/utils/losses.py#L154) ###
    logits_io = torch.mm(zi_norm, zo_norm.t()) / temp
    logits_jo = torch.mm(zj_norm, zo_norm.t()) / temp
    probs_io_zi = F.softmax(logits_io[torch.logical_not(mask)], -1)
    probs_jo_zj = F.log_softmax(logits_jo[torch.logical_not(mask)], -1)
    SIR_loss = F.kl_div(probs_io_zi, probs_jo_zj, log_target=True, reduction="sum")

    ### Adversarial Invariant Regularization ###
    logits_io = torch.mm(zi_adv_norm, zi_norm.t()) / temp
    logits_jo = torch.mm(zj_adv_norm, zj_norm.t()) / temp
    probs_io_zi_adv_consis = F.softmax(logits_io[torch.logical_not(mask)], -1)
    probs_jo_zj_adv_consis = F.softmax(logits_jo[torch.logical_not(mask)], -1)

    logits_io = torch.mm(zi_adv_norm, zo_norm.t()) / temp
    logits_jo = torch.mm(zj_adv_norm, zo_norm.t()) / temp
    probs_io_zi_adv = F.softmax(logits_io[torch.logical_not(mask)], -1)
    probs_jo_zj_adv = F.softmax(logits_jo[torch.logical_not(mask)], -1)

    probs_io_zi_adv = torch.mul(probs_io_zi_adv, probs_io_zi_adv_consis)
    probs_jo_zj_adv = torch.mul(probs_jo_zj_adv, probs_jo_zj_adv_consis)
    AIR_loss = F.kl_div(probs_io_zi_adv, torch.log(probs_jo_zj_adv), log_target=True, reduction="sum")


    return SIR_loss, AIR_loss


def reg_loss_(zo_norm, zi_norm, zi_adv_norm, zj_norm, zj_adv_norm, temp=0.5):


    bs = zi_norm.shape[0]
    mask = torch.ones((bs, bs), dtype=bool).fill_diagonal_(0)

    ### Standard Invariant Regularization (Implementation follows https://github.com/NightShade99/Self-Supervised-Vision/blob/42183d391f51c60383ff0cb044e2d71379aa7461/utils/losses.py#L154) ###
    logits_io = torch.mm(zi_norm, zo_norm.t()) / temp
    logits_jo = torch.mm(zj_norm, zo_norm.t()) / temp
    # get log probabilities
    log_probs_io_zi = F.log_softmax(logits_io[torch.logical_not(mask)], -1)
    log_probs_jo_zj = F.log_softmax(logits_jo[torch.logical_not(mask)], -1)
    SIR_loss = F.kl_div(log_probs_io_zi, log_probs_jo_zj, log_target=True, reduction="sum")


    ### Adversarial Invariant Regularization ###
    logits_io = torch.mm(zi_adv_norm, zi_norm.t()) / temp
    logits_jo = torch.mm(zj_adv_norm, zj_norm.t()) / temp
    probs_io_zi_adv_consis = F.softmax(logits_io[torch.logical_not(mask)], -1)
    probs_jo_zj_adv_consis = F.softmax(logits_jo[torch.logical_not(mask)], -1)

    logits_io = torch.mm(zi_adv_norm, zo_norm.t()) / temp
    logits_jo = torch.mm(zj_adv_norm, zo_norm.t()) / temp
    probs_io_zi_adv = F.softmax(logits_io[torch.logical_not(mask)], -1)
    probs_jo_zj_adv = F.softmax(logits_jo[torch.logical_not(mask)], -1)

    probs_io_zi_adv = torch.mul(probs_io_zi_adv, probs_io_zi_adv_consis)
    probs_jo_zj_adv = torch.mul(probs_jo_zj_adv, probs_jo_zj_adv_consis)
    # get log probabilities
    log_probs_io_zi_adv = F.log_softmax(probs_io_zi_adv, -1)
    log_probs_jo_zj_adv = F.log_softmax(probs_jo_zj_adv, -1)
    AIR_loss = F.kl_div(log_probs_io_zi_adv, log_probs_jo_zj_adv, log_target=True, reduction="sum")



    return SIR_loss, AIR_loss
