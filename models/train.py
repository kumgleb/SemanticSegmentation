import numpy as np
from tqdm import tqdm
from IPython.display import clear_output
from utils.train_utils import train_monitor


import torch
from torch.utils.tensorboard import SummaryWriter


def forward(model, x, y, criterion, device):
    x = x.to(device)
    y = y.to(device)
    prd = model(x)
    loss = criterion(prd, y)
    return loss, prd


def evaluate(model, dataloader, device, criterion):
    losses = []
    model.eval()
    with torch.no_grad():
        val_iter = iter(dataloader)
        for i in range(len(dataloader)):
            x, y = next(val_iter)
            loss, _ = forward(model, x, y, criterion, device)
            losses.append(loss.item())
    return np.mean(losses)


def train_model(model,
                train_dataloader,
                val_dataloader,
                device,
                optimizer,
                criterion,
                scheduler,
                cfg):

    losses_train, losses_train_mean = [], []
    losses_val = []
    loss_val = None
    best_val_loss = 1e6

    writer = SummaryWriter()

    tr_it = iter(train_dataloader)
    progress_bar = tqdm(range(cfg['n_iter']))

    for i in progress_bar:
        try:
            x, y = next(tr_it)
        except StopIteration:
            tr_it = iter(train_dataloader)
            x, y = next(tr_it)

        model.train()
        torch.set_grad_enabled(True)
        loss, _ = forward(model, x, y, criterion, device)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses_train.append(loss.item())
        losses_train_mean.append(np.mean(losses_train[-1:-10:-1]))
        progress_bar.set_description(f'loss: {loss.item():.5f}, avg loss: {np.mean(losses_train):.5f}')
        writer.add_scalar('loss_train', loss.item(), i)

        if i % cfg['n_iter_val'] == 0:
            loss_val = evaluate(model, val_dataloader, device, criterion)
            losses_val.append(loss_val)
            progress_bar.set_description(f'val_loss: {loss_val:.5f}')
            writer.add_scalar('loss_val', loss_val, i)

        if scheduler:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau' and loss_val:
                scheduler.step(loss_val)
            else:
                scheduler.step()
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], i)

        clear_output(True)
        if cfg['plot_mode']:
            train_monitor(losses_train, losses_train_mean, losses_val)

        if cfg['save_best_val'] and loss_val < best_val_loss:
            best_val_loss = loss_val
            checkpoint_path = cfg['checkpoint_path']
            torch.save(model.state_dict(),
                       f'{checkpoint_path}/{model.__class__.__name__}_{loss_val:.3f}.pth')

    writer.close()

