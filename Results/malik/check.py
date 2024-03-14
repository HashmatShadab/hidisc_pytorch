import torch

if __name__ == "__main__":
    ckpt = torch.load("best_clean_loss_checkpoint.pth")
    print(ckpt["epoch"])
    print(ckpt["loss"])
    ckpt = torch.load("best_adv_loss_checkpoint.pth")
    print(ckpt["epoch"])
    print(ckpt["loss"])
