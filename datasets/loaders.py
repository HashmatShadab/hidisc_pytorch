from .improc import get_srh_aug_list
from .srh_dataset import HiDiscDataset
import os
from functools import partial
from torchvision.transforms import Compose
import torch


def get_dataloaders(cf):
    """Create dataloader for contrastive experiments."""
    train_dset = HiDiscDataset(
        data_root=cf["data"]["db_root"],
        studies="train",
        transform=Compose(get_srh_aug_list(cf["data"]["train_augmentation"])),
        balance_study_per_class=cf["data"]["balance_study_per_class"],
        num_slide_samples=cf["data"]["hidisc"]["num_slide_samples"],
        num_patch_samples=cf["data"]["hidisc"]["num_patch_samples"],
        num_transforms=cf["data"]["hidisc"]["num_transforms"])
    val_dset = HiDiscDataset(
        data_root=cf["data"]["db_root"],
        studies="val",
        transform=Compose(get_srh_aug_list(cf["data"]["valid_augmentation"])),
        balance_study_per_class=False,
        num_slide_samples=cf["data"]["hidisc"]["num_slide_samples"],
        num_patch_samples=cf["data"]["hidisc"]["num_patch_samples"],
        num_transforms=cf["data"]["hidisc"]["num_transforms"])

    dataloader_callable = partial(torch.utils.data.DataLoader,
                                  batch_size=cf['training']['batch_size'],
                                  drop_last=False,
                                  pin_memory=True,
                                  num_workers=get_num_worker(),
                                  persistent_workers=True)

    return dataloader_callable(train_dset,
                               shuffle=True), dataloader_callable(val_dset,
                                                                  shuffle=True)




def get_num_worker():
    """Estimate number of cpu workers."""
    try:
        num_worker = len(os.sched_getaffinity(0))
    except Exception:
        num_worker = os.cpu_count()

    if num_worker > 1:
        return num_worker - 1
    else:
        return torch.cuda.device_count() * 4
