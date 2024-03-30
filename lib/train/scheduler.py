from collections import Counter
from lib.utils.optimizer.lr_scheduler import WarmupMultiStepLR, MultiStepLR, ExponentialLR
import torch.optim as optim

def make_lr_scheduler(cfg, optimizer):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        scheduler = MultiStepLR(optimizer,
                                milestones=cfg_scheduler.milestones,
                                gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'exponential':
        scheduler = ExponentialLR(optimizer,
                                  decay_epochs=cfg_scheduler.decay_epochs,
                                  gamma=cfg_scheduler.gamma)
    elif cfg_scheduler.type == 'one_cycle':
        scheduler = optim.lr_scheduler.OneCycleLR(optimizer,
                                                  cfg.train.lr,
                                                  cfg.train.epoch,
                                                  pct_start=0.1,
                                                  cycle_momentum=False,
                                                  anneal_strategy="linear")

    return scheduler


def set_lr_scheduler(cfg, scheduler):
    cfg_scheduler = cfg.train.scheduler
    if cfg_scheduler.type == 'multi_step':
        scheduler.milestones = Counter(cfg_scheduler.milestones)
    elif cfg_scheduler.type == 'exponential':
        scheduler.decay_epochs = cfg_scheduler.decay_epochs
    elif cfg_scheduler.type == 'one_cycle':
        scheduler.max_lr = cfg.train.lr
        scheduler.total_step = cfg.train.epoch
    scheduler.gamma = cfg_scheduler.gamma
