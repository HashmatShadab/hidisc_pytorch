import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import random



def pgd_attack(model, images, labels, device, eps=8. / 255., alpha=2. / 255., iters=20, advFlag=None, forceEval=True, randomInit=True):
    # images = images.to(device)
    # labels = labels.to(device)
    loss = nn.CrossEntropyLoss()

    # init
    if randomInit:
        delta = torch.rand_like(images) * eps * 2 - eps
    else:
        delta = torch.zeros_like(images)
    delta = torch.nn.Parameter(delta, requires_grad=True)

    for i in range(iters):
        if advFlag is None:
            if forceEval:
                model.eval()
            outputs = model(images + delta)
        else:
            if forceEval:
                model.eval()
            outputs = model(images + delta, advFlag)

        model.zero_grad()
        cost = loss(outputs, labels)
        # cost.backward()
        delta_grad = torch.autograd.grad(cost, [delta])[0]

        delta.data = delta.data + alpha * delta_grad.sign()
        delta.grad = None
        delta.data = torch.clamp(delta.data, min=-eps, max=eps)
        delta.data = torch.clamp(images + delta.data, min=0, max=1) - images

    model.zero_grad()

    return (images + delta).detach()

def trades_loss(model, x_natural, y, optimizer, step_size=2/255, epsilon=8/255, perturb_steps=10, beta=6.0, distance='l_inf'):
    batch_size = len(x_natural)
    # define KL-loss
    criterion_kl = nn.KLDivLoss(size_average=False)
    model.eval()

    # generate adversarial example
    x_adv = x_natural.detach()
    if distance == 'l_inf':
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                model.eval()
                loss_kl = criterion_kl(F.log_softmax(model(x_adv), dim=1),
                                       F.softmax(model(x_natural), dim=1))
            grad = torch.autograd.grad(loss_kl, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, x_natural -
                              epsilon), x_natural + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
    else:
        assert False

    x_adv = Variable(torch.clamp(x_adv, 0.0, 1.0), requires_grad=False)

    # zero gradient
    optimizer.zero_grad()
    model.train()
    # calculate robust loss
    logits = model(x_natural)

    loss = F.cross_entropy(logits, y)

    logits_adv = model(x_adv)

    loss_robust = (1.0 / batch_size) * criterion_kl(F.log_softmax(logits_adv, dim=1),
                                                    F.softmax(logits, dim=1))
    loss += beta * loss_robust

    return loss


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
