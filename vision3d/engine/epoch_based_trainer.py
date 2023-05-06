import abc

import torch

from vision3d.utils.misc import get_log_string
from vision3d.utils.summary_board import SummaryBoard
from vision3d.utils.tensor import move_to_cuda
from vision3d.utils.timer import Timer

from .base_trainer import BaseTrainer
from .context_manager import clear_context_manager


class EpochBasedTrainer(BaseTrainer, abc.ABC):
    """Epoch-based Trainer.

    The training lasts for 'cfg.trainer.max_epoch' epochs, and each epoch goes through the whole training set.

    The learning rate is decayed after each epoch.

    Training pipeline:
        1. before_train_epoch
        2. for each iteration:
            2.1 before_train_step
            2.2 train_step
            2.3 after_backward
            2.4 optimizer_step
            2.5 after_train_step
            2.6 log iteration
        3. scheduler_step
        4. after_train_epoch
    """

    def __init__(self, cfg):
        super().__init__(cfg)

    def train_epoch(self):
        """Training epoch."""
        # before train epoch
        if hasattr(self.train_loader.sampler, "set_epoch"):
            self.train_loader.sampler.set_epoch(self.epoch)
        self.before_train_epoch(self.epoch)
        # setup watcher
        summary_board = SummaryBoard(auto_register=True)
        timer = Timer()
        # train loop
        self.optimizer.zero_grad()
        max_iteration = len(self.train_loader)
        timer.tic("data")
        for batch_index, data_dict in enumerate(self.train_loader):
            clear_context_manager()
            # load data
            self.iteration = batch_index + 1
            self.total_steps += 1
            data_dict = move_to_cuda(data_dict)
            self.before_train_step(self.epoch, self.iteration, data_dict)
            timer.toc("data")
            # forward
            timer.tic("model")
            output_dict, result_dict = self.train_step(self.epoch, self.iteration, data_dict)
            if "loss" in result_dict:
                # backward
                result_dict["loss"].backward()
                self.after_backward(self.epoch, self.iteration, data_dict, output_dict, result_dict)
                self.check_gradients(self.epoch, self.iteration, data_dict, output_dict, result_dict)
                # optimization
                if self.iteration % self.grad_acc_steps == 0:
                    self.optimizer_step()
            timer.toc("model")
            # after training
            timer.tic("data")
            self.after_train_step(self.epoch, self.iteration, data_dict, output_dict, result_dict)
            result_dict = self.unpack_tensors(result_dict)
            summary_board.update_from_dict(result_dict)
            # logging
            if self.iteration % self.log_steps == 0:
                summary_dict = summary_board.summary(last_n=self.log_steps)
                message = get_log_string(
                    summary_dict,
                    epoch=self.epoch,
                    max_epoch=self.max_epoch,
                    iteration=self.iteration,
                    max_iteration=max_iteration,
                    lr=self.get_lr(),
                    time_dict=timer.summary(keys=["data", "model"]),
                )
                self.log(message)
                self.write_event("train", summary_dict, self.total_steps)
            # empty cache
            torch.cuda.empty_cache()
        # scheduler
        self.scheduler_step()
        # after train epoch
        summary_dict = summary_board.summary()
        self.after_train_epoch(self.epoch, summary_dict)
