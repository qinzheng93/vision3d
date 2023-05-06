import abc
import json
import os.path as osp
import time
from typing import Tuple

import ipdb
import torch
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

from vision3d.utils.distributed import (
    all_reduce_tensors,
    get_local_rank,
    get_world_size,
    is_distributed,
    setup_distributed,
)
from vision3d.utils.io import ensure_dir
from vision3d.utils.logger import get_logger
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import get_default_parser
from vision3d.utils.summary_board import SummaryBoard
from vision3d.utils.tensor import move_to_cuda, tensor_to_array
from vision3d.utils.timer import Timer

from .checkpoint import CheckpointManager
from .context_manager import clear_context_manager
from .utils import setup_engine


class BaseTrainer(abc.ABC):
    """Base Trainer.

    1. Usages:
        1.1 Use 'register_model' to add the model to the trainer.
        1.2 Use 'register_optimizer' to add the optimizer to the trainer.
        1.3 Use 'register_scheduler' to add the scheduler to the trainer.
        1.4 Use 'register_loader' to add the training and validation data loaders to the trainer.
        1.6 Use 'save_best_model_on_metric' to save best models during training.
        1.7 The 'train_step' and 'val_step' methods must be implemented. Other methods can be optionally implemented.

    2. Pseudo Batch:
        Use 'cfg.trainer.grad_acc_steps' pseudo batching. Every 'cfg.trainer.grad_acc_steps' iterations are used as a
        pseudo batch. And the gradients in a pseudo batch are accumulated during backward propagation. Note that the
        gradients are NOT averaged in the pseudo batch.

    3. About Multi-GPU training.
        Multi-GPU training is supported by 'DistributedDataParallel' and the trainer automatically enables it when
        launched with `torchrun`. Only one GPU (cuda:0) will be used otherwise even if there are multiple GPUs. Note
        that the learning rate is scaled by the world size by default.

    4. Validation pipeline:
        1. before_val_epoch
        2. for each iteration:
            2.1 before_val_step
            2.2 val_step
            2.3 after_val_step
            2.4 log iteration
        3. summary logging
        4. update best models
        5. after_val_epoch

    5. Checkpointing:
        5.1 After each training epoch, the metadata (epoch, total_steps) and the weights are saved to 'epoch-N.pth'.
        5.2 After each validation epoch, the metadata, the weights and the training states (optimizer, scheduler and
            checkpoint manager) are saved to 'checkpoint.pth'. If the training is interrupted, use '--resume' to resume
            training from the last finished validation epoch.
    """

    def __init__(self, cfg):
        # parser
        parser = get_default_parser()
        self._args = parser.parse_args()
        self._debug_mode = self._args.debug
        self._detect_anomaly = self._args.detect_anomaly
        self._cudnn_deterministic = self._args.cudnn_deterministic
        self._save_latest_n_models = self._args.save_latest_n_models
        self._log_steps = self._args.log_steps

        # cuda check
        assert torch.cuda.is_available(), "No CUDA devices available."

        # distributed data parallel
        setup_distributed()
        self._is_distributed = is_distributed()
        self._world_size = get_world_size()
        self._local_rank = get_local_rank()

        # parse config
        self._cfg = cfg
        self._output_dir = cfg.exp.output_dir
        self._checkpoint_dir = cfg.exp.checkpoint_dir
        self._log_dir = cfg.exp.log_dir
        self._event_dir = cfg.exp.event_dir
        self._manual_seed = cfg.exp.seed + self._local_rank
        self._max_epoch = cfg.trainer.max_epoch
        self._grad_acc_steps = cfg.trainer.grad_acc_steps

        # logger
        self._log_file = osp.join(self._log_dir, "train-{}.log".format(time.strftime("%Y%m%d-%H%M%S")))
        self._logger = get_logger(log_file=self._log_file, event_dir=self._event_dir)

        # print info
        self.log("Configs:\n" + json.dumps(self._cfg, indent=4))
        if self._is_distributed:
            self.log(f"Using DistributedDataParallel mode (world_size: {self._world_size})")
        else:
            self.log("Using Single-GPU mode.")

        # initialization
        setup_engine(seed=self._manual_seed, cudnn_deterministic=self._cudnn_deterministic, debug=self._detect_anomaly)

        # checkpoint manager
        self._ckpt_manager = CheckpointManager(self._checkpoint_dir, save_latest_models=self._save_latest_n_models)

        # find checkpoint
        self._checkpoint = osp.join(self._checkpoint_dir, "checkpoint.pth") if self._args.resume else None

        # state
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.epoch = 0
        self.iteration = 0
        self.total_steps = 0
        self.train_loader = None
        self.val_loader = None

    @property
    def args(self):
        return self._args

    @property
    def log_steps(self):
        return self._log_steps

    @property
    def max_epoch(self):
        return self._max_epoch

    @property
    def grad_acc_steps(self):
        return self._grad_acc_steps

    @property
    def metadata(self):
        """Metadata (epoch, total_steps)."""
        metadata = {"epoch": self.epoch, "total_steps": self.total_steps}
        return metadata

    def save(self, filename, save_training_states=False):
        """Save checkpoint."""
        if not save_training_states:
            # save model only, used after each training epoch
            self._ckpt_manager.save_checkpoint(
                filename,
                self.metadata,
                self.model,
                optimizer=None,
                scheduler=None,
                save_ckpt_manager=False,
                clean_checkpoints=True,
            )
        else:
            # save checkpoint with optimizer, scheduler and ckpt_manager, used after each validation epoch
            self._ckpt_manager.save_checkpoint(
                filename,
                self.metadata,
                self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                save_ckpt_manager=True,
                clean_checkpoints=False,
            )

    def load(self, filename):
        """Load checkpoint."""
        # load model, optimizer and scheduler
        metadata = self._ckpt_manager.load_checkpoint(
            filename, self.model, optimizer=self.optimizer, scheduler=self.scheduler
        )
        # Load metadata (epoch, total_steps)
        self.epoch = metadata["epoch"]
        self.total_steps = metadata["total_steps"]
        self.log(f"Checkpoint metadata: epoch: {self.epoch}, total_steps: {self.total_steps}.")

    def update_best_models(self, metric_dict):
        """Update best models."""
        self._ckpt_manager.update_best_model(metric_dict, self.metadata, self.model)

    def save_best_model_on_metric(self, metric, largest=True):
        """Set metric to save best model."""
        self._ckpt_manager.add_metric(metric, largest=largest)

    def register_model(self, model):
        """Register model. DDP is automatically used."""
        model = model.cuda()
        if self._is_distributed:
            model = DistributedDataParallel(model, device_ids=[self._local_rank], output_device=self._local_rank)
        self.model = model
        self.log("Model description:\n" + str(model))
        return model

    def register_optimizer(self, optimizer):
        """Register optimizer. DDP is automatically used."""
        if self._is_distributed:
            for param_group in optimizer.param_groups:
                param_group["lr"] = param_group["lr"] * self._world_size
        self.optimizer = optimizer

    def register_scheduler(self, scheduler):
        """Register LR scheduler."""
        self.scheduler = scheduler

    def register_loader(self, train_loader, val_loader):
        """Register data loader."""
        self.train_loader = train_loader
        self.val_loader = val_loader

    def get_lr(self):
        """Get the learning rate for the first param group."""
        return self.optimizer.param_groups[0]["lr"]

    def set_train_mode(self):
        """Set training mode."""
        self.model.train()
        torch.set_grad_enabled(True)

    def set_eval_mode(self):
        """Set evaluation mode."""
        self.model.eval()
        torch.set_grad_enabled(False)

    def log(self, message, level="INFO"):
        """Logger wrapper."""
        self._logger.log(message, level=level)

    def write_event(self, phase, event_dict, index):
        """Write TensorBoard event."""
        for key, value in event_dict.items():
            self._logger.write_event(f"{phase}/{key}", value, index)

    def unpack_tensors(self, result_dict):
        """All reduce and release tensors."""
        if self._is_distributed:
            result_dict = all_reduce_tensors(result_dict, world_size=self._world_size)
        result_dict = tensor_to_array(result_dict)
        return result_dict

    def optimizer_step(self):
        """Optimizer step."""
        self.optimizer.step()
        self.optimizer.zero_grad()

    def scheduler_step(self):
        """LR Scheduler step."""
        if self.scheduler is not None:
            self.scheduler.step()

    def before_train(self):
        """Before training function."""
        pass

    def before_train_epoch(self, epoch):
        """Before training epoch function."""
        pass

    def before_train_step(self, epoch, iteration, data_dict):
        """Before training step function."""
        pass

    @abc.abstractmethod
    def train_step(self, epoch, iteration, data_dict) -> Tuple[dict, dict]:
        """Training step. Must be implemented."""
        pass

    def after_backward(self, epoch, iteration, data_dict, output_dict, result_dict):
        """After backward propagation function."""
        pass

    def after_train_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        """After training step function."""
        pass

    def after_train_epoch(self, epoch, summary_dict):
        """After training epoch function."""
        pass

    def before_val_epoch(self, epoch):
        """Before validation epoch function."""
        pass

    def before_val_step(self, epoch, iteration, data_dict):
        """Before validation step function."""
        pass

    @abc.abstractmethod
    def val_step(self, epoch, iteration, data_dict) -> Tuple[dict, dict]:
        """Validation step. Must be implemented."""
        pass

    def after_val_step(self, epoch, iteration, data_dict, output_dict, result_dict):
        """After validation step function."""
        pass

    def after_val_epoch(self, epoch, summary_dict):
        """After validation epoch function."""
        pass

    def check_gradients(self, epoch, iteration, data_dict, output_dict, result_dict):
        """Check if any gradients are NaN or Inf."""
        if not self._debug_mode:
            return

        all_gradients_ok = True
        for param in self.model.parameters():
            if not torch.all(torch.isfinite(param.grad)):
                all_gradients_ok = False
                break

        if all_gradients_ok:
            return

        self.log("Epoch: {}, iter: {}, gradients.".format(epoch, iteration), level="ERROR")
        debug_dir = osp.join(self._output_dir, "debug")
        ensure_dir(debug_dir)
        data_file = osp.join(debug_dir, "data.pth")
        torch.save(data_dict, data_file)
        self.log(f"Data dict saved to '{data_file}'.", level="ERROR")
        model_file = osp.join(debug_dir, "model.pth")
        torch.save(self.model, model_file)
        self.log(f"Model object saved to '{model_file}'.", level="ERROR")

        ipdb.set_trace()

    @abc.abstractmethod
    def train_epoch(self):
        """Training epoch."""
        pass

    def val_epoch(self):
        """Validation epoch."""
        # before val epoch
        self.set_eval_mode()
        self.before_val_epoch(self.epoch)
        # setup watcher
        summary_board = SummaryBoard(auto_register=True)
        timer = Timer()
        # val loop
        max_iteration = len(self.val_loader)
        pbar = tqdm(enumerate(self.val_loader), total=max_iteration, disable=self._local_rank != 0)
        timer.tic("data")
        for iteration, data_dict in pbar:
            clear_context_manager()
            # load data
            self.iteration = iteration + 1
            data_dict = move_to_cuda(data_dict)
            self.before_val_step(self.epoch, self.iteration, data_dict)
            timer.toc("data")
            # val step
            torch.cuda.synchronize()
            timer.tic("model")
            output_dict, result_dict = self.val_step(self.epoch, self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.toc("model")
            # after val step
            timer.tic("data")
            self.after_val_step(self.epoch, self.iteration, data_dict, output_dict, result_dict)
            result_dict = self.unpack_tensors(result_dict)
            summary_board.update_from_dict(result_dict)
            # logging
            message = get_log_string(
                summary_board.summary(),
                epoch=self.epoch,
                max_epoch=self._max_epoch,
                iteration=self.iteration,
                max_iteration=max_iteration,
                time_dict=timer.summary(keys=["data", "model"]),
            )
            pbar.set_description(message)
            torch.cuda.empty_cache()
        # summary logging
        summary_dict = summary_board.summary()
        message = get_log_string(summary_dict, epoch=self.epoch, time_dict=timer.summary(keys=["data", "model"]))
        self.log("[Val] " + message, level="SUCCESS")
        self.write_event("val", summary_dict, self.epoch)
        # after val epoch
        self.after_val_epoch(self.epoch, summary_dict)
        self.set_train_mode()
        # update model
        self.update_best_models(summary_dict)

    def debug(self):
        pass

    def run(self, only_validation=False, debug=False):
        """Run trainer."""
        # before train check
        assert self.train_loader is not None, "No training loader found."
        assert self.val_loader is not None, "No validation loader found."
        assert len(self.train_loader) > 0, "Training loader is empty."
        assert len(self.val_loader) > 0, "Validation loader is empty."
        assert self.model is not None, "No model found."
        assert self.optimizer is not None, "No optimizer found."
        if self.scheduler is None:
            self.log("LRScheduler is not set.", level="WARNING")

        # load checkpoint
        if self._checkpoint is not None:
            self.load(self._checkpoint)

        # before train
        self.set_train_mode()
        self.before_train()

        if debug:
            self.debug()

        # run training
        while self.epoch < self._max_epoch:
            self.epoch += 1

            if not only_validation:
                self.train_epoch()
                self.save(f"epoch-{self.epoch}.pth", save_training_states=False)

            self.val_epoch()
            self.save("checkpoint.pth", save_training_states=True)
