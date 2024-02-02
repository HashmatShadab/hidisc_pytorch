import torch
from torch.cuda.amp import autocast
from helpers import  accuracy, MetricLogger, plot_grid


def validate_clean(val_loader, model, criterion):
    # Distributed metric logger
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'

    # Switch to evaluation mode
    model.eval()

    # Evaluate and return the loss and accuracy
    with torch.no_grad():
        for images, targets in metric_logger.log_every(val_loader, 10, header):
            # Move the tensors to the GPUs
            images = images.to("cuda", non_blocking=True)
            targets = targets.to("cuda", non_blocking=True)

            # Forward pass to the network
            outputs, final_inp = model(images, target=targets, make_adv=False)


            # Calculate loss
            loss = criterion(outputs, targets)

            # Measure accuracy
            acc1 = accuracy(outputs, targets, topk=(1,))[0]
            acc5 = accuracy(outputs, targets, topk=(5,))[0]

            # Update the losses & top1 accuracy list
            batch_size = images.shape[0]
            metric_logger.update(loss=loss.item())
            metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
            metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    # Gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def evaluate_standard_accuracy(model, val_loader, criterion, args):

    """
    Results will be evaluated on full size image, no ablated input will be passed, so no tokens will be dropped
    """

    if args.distributed:
        model.module.normalizer.do_ablation = False
        model.module.normalizer.return_mask = False
    else:
        model.normalizer.do_ablation = False
        model.normalizer.return_mask = False

    test_stats_clean = validate_clean(val_loader=val_loader, model=model,
                                      criterion=criterion)

    if args.distributed:
        model.module.normalizer.do_ablation = True
        model.module.normalizer.return_mask = True
    else:
        model.normalizer.do_ablation = True
        model.normalizer.return_mask = True

    return test_stats_clean
