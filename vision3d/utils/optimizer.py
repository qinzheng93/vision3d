from typing import Callable

import numpy as np
import torch
import torch.optim as optim
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from .misc import deprecated


class WarmUpExponentialAnnealingFunction(Callable):
    def __init__(self, warmup_steps: int, gamma: float, step_size: int, eta_init: float = 0.1, eta_min: float = 0.1):
        self.warmup_steps = warmup_steps
        self.gamma = gamma
        self.step_size = step_size
        self.eta_init = eta_init
        self.eta_min = eta_min

    def __call__(self, last_step: int) -> float:
        # last_step starts from -1, which means next_step=0 indicates the first call of lr annealing.
        next_step = last_step + 1
        # warm up
        if next_step < self.warmup_steps:
            return self.eta_init + (1.0 - self.eta_init) / self.warmup_steps * next_step
        # exponential decay
        decay_step = next_step - self.warmup_steps
        return max(self.gamma ** ((decay_step + 1) // self.step_size), self.eta_min)


class WarmUpCosineAnnealingFunction(Callable):
    def __init__(self, total_steps: int, warmup_steps: int, eta_init: float = 0.1, eta_min: float = 0.1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = total_steps - warmup_steps
        self.eta_init = eta_init
        self.eta_min = eta_min

    def __call__(self, last_step: int) -> float:
        # last_step starts from -1, which means next_step=0 indicates the first call of lr annealing.
        next_step = last_step + 1
        # finish decay
        if next_step > self.total_steps:
            return self.eta_min
        # warm up
        if next_step < self.warmup_steps:
            return self.eta_init + (1.0 - self.eta_init) / self.warmup_steps * next_step
        # cosine decay
        decay_step = next_step - self.warmup_steps
        return self.eta_min + 0.5 * (1.0 - self.eta_min) * (1 + np.cos(np.pi * decay_step / self.decay_steps))


class WarmUpLinearAnnealingFunction(Callable):
    def __init__(self, total_steps: int, warmup_steps: int, eta_init: float = 0.1, eta_min: float = 0.1):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_steps = total_steps - warmup_steps
        self.eta_init = eta_init
        self.eta_min = eta_min

    def __call__(self, last_step: int) -> float:
        # last_step starts from -1, which means next_step=0 indicates the first call of lr annealing.
        next_step = last_step + 1
        # finish decay
        if next_step > self.total_steps:
            return self.eta_min
        # warm up
        if next_step < self.warmup_steps:
            return self.eta_init + (1.0 - self.eta_init) / self.warmup_steps * next_step
        # linear decay
        decay_step = next_step - self.warmup_steps
        return self.eta_min + (1.0 - self.eta_min) * (1.0 - decay_step / self.decay_steps)


def build_warmup_lr_scheduler(optimizer: Optimizer, cfg, annealing: str = "cosine") -> _LRScheduler:
    assert annealing in ["cosine", "linear", "exponential"]

    grad_acc_steps = cfg.trainer.grad_acc_steps
    warmup_steps = cfg.scheduler.warmup_steps // grad_acc_steps
    eta_init = cfg.scheduler.eta_init
    eta_min = cfg.scheduler.eta_min

    if annealing == "exponential":
        gamma = cfg.scheduler.gamma
        step_size = cfg.scheduler.step_size
        annealing_func = WarmUpExponentialAnnealingFunction(
            warmup_steps, gamma, step_size, eta_init=eta_init, eta_min=eta_min
        )
    else:
        total_steps = cfg.scheduler.total_steps // grad_acc_steps
        if annealing == "cosine":
            annealing_func = WarmUpCosineAnnealingFunction(
                total_steps, warmup_steps, eta_init=eta_init, eta_min=eta_min
            )
        else:
            annealing_func = WarmUpLinearAnnealingFunction(
                total_steps, warmup_steps, eta_init=eta_init, eta_min=eta_min
            )

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, annealing_func)

    return scheduler


def build_optimizer(model: Module, cfg) -> Optimizer:
    """Build optimizer.

    The optimizer type is determined by "cfg.optim.optimizer".

    Available optimizers are: "SGD", "Adam" and "AdamW".

    1. "SGD" parameters:
        "cfg.optim.lr": The learning rate.
        "cfg.optim.momentum": The SGD momentum.
        "cfg.optim.weight_decay": The weight decay for L2 regularization.

    2. "Adam" parameters:
        "cfg.optim.lr": The learning rate.
        "cfg.optim.weight_decay": The weight decay for L2 regularization.

    3. "AdamW" parameters:
        The same as "Adam".

    Args:
        model (Module): The parameters for the optimizer.
        cfg (EasyDict): The config.

    Returns:
        optimizer (Optimizer): The optimizer built.
    """
    params = model.parameters()
    optim_type = cfg.optimizer.type
    if optim_type == "SGD":
        optimizer = optim.SGD(
            params, lr=cfg.optimizer.lr, momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay
        )
    elif optim_type == "Adam":
        optimizer = optim.Adam(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    elif optim_type == "AdamW":
        optimizer = optim.AdamW(params, lr=cfg.optimizer.lr, weight_decay=cfg.optimizer.weight_decay)
    else:
        raise RuntimeError(f"Unsupported optimizer: {optim_type}. Supported optimizers: 'SGD', 'Adam', 'AdamW'.")
    return optimizer


def build_scheduler(optimizer: Optimizer, cfg) -> _LRScheduler:
    """Build lr scheduler.

    The scheduler type is determined by "cfg.optim.scheduler".

    Available schedulers are: None, "Step", "Exponential", "Cosine", "Linear".

    1. "Step" parameters:
        "gamma": The decay factor.
        "step_size": The step size for lr decay.

    2. "Cosine" configuration:
        "total_steps": The total iterations to run.
        "warmup_steps": The first steps are used for linear warmup.
        "eta_init": The starting lr scaling factor for warmup.
        "eta_min": The minimal lr scaling factor during decay.

    3. "Linear" configuration:
        The same as "Cosine".

    4. "Exponential" configuration:
        "warmup_steps": The first steps are used for linear warmup.
        "gamma": The decay factor.
        "step_size": The step size for lr decay.
        "eta_init": The starting lr scaling factor for warmup.
        "eta_min": The minimal lr scaling factor during decay.


    Args:
        optimizer (Optimizer): The optimizer for the scheduler.
        cfg (EasyDict): The config.

    Returns:
        scheduler (_LRScheduler): The scheduler built.
    """
    sched_type = cfg.scheduler.type
    if sched_type is None or sched_type == "None":
        scheduler = None
    elif sched_type == "Step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=cfg.scheduler.step_size, gamma=cfg.scheduler.gamma)
    elif sched_type == "Exponential":
        scheduler = build_warmup_lr_scheduler(optimizer, cfg, annealing="exponential")
    elif sched_type == "Cosine":
        scheduler = build_warmup_lr_scheduler(optimizer, cfg, annealing="cosine")
    elif sched_type == "Linear":
        scheduler = build_warmup_lr_scheduler(optimizer, cfg, annealing="linear")
    else:
        raise RuntimeError(
            f"Unsupported scheduler: {sched_type}. Supported optimizers: 'Step', 'Exponential', 'Cosine', 'Linear'."
        )

    return scheduler


@deprecated("build_scheduler")
def build_warmup_cosine_lr_scheduler(optimizer, total_steps, warmup_steps, eta_init=0.1, eta_min=0.1, grad_acc_steps=1):
    total_steps //= grad_acc_steps
    warmup_steps //= grad_acc_steps
    cosine_func = WarmUpCosineAnnealingFunction(total_steps, warmup_steps, eta_init=eta_init, eta_min=eta_min)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, cosine_func)
    return scheduler
