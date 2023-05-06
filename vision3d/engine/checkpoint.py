import os
import os.path as osp

import torch
from torch.nn.parallel import DistributedDataParallel

from vision3d.utils.distributed import master_only
from vision3d.utils.logger import get_logger


def load_state_dict(model, state_dict, strict=False):
    logger = get_logger()

    if isinstance(model, DistributedDataParallel):
        model = model.module

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if len(missing_keys) > 0:
        logger.warn(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        logger.warn(f"Unexpected keys: {unexpected_keys}")

    if strict and (len(missing_keys) != 0 or len(unexpected_keys) != 0):
        raise RuntimeError("The keys in model and state_dict do not match.")


class CheckpointManager:
    """Checkpoint Manager.

    Note:
        If the program stopped during validation, the last epoch will be ignored during model selection. Resuming
        training will fix this.
    """

    def __init__(self, checkpoint_dir, save_latest_models=-1):
        self._checkpoint_dir = checkpoint_dir
        self._save_latest_models = save_latest_models
        self._saved_checkpoints = []
        self._best_scores = {}

    def state_dict(self):
        return {"saved_checkpoints": self._saved_checkpoints, "best_scores": self._best_scores}

    def load_state_dict(self, state_dict):
        self._saved_checkpoints = state_dict["saved_checkpoints"]
        self._best_scores = state_dict["best_scores"]

    @master_only
    def save_checkpoint(
        self, filename, metadata, model, optimizer=None, scheduler=None, save_ckpt_manager=False, clean_checkpoints=True
    ):
        logger = get_logger()

        # handle DDP model
        if isinstance(model, DistributedDataParallel):
            model = model.module

        state_dict = {"metadata": metadata, "model": model.state_dict()}

        # optimizer, scheduler
        if optimizer is not None:
            state_dict["optimizer"] = optimizer.state_dict()
        if scheduler is not None:
            state_dict["scheduler"] = scheduler.state_dict()

        # checkpoint manager
        if save_ckpt_manager:
            state_dict["ckpt_manager"] = self.state_dict()

        # NOTE: save only model weights and metadata
        filename = osp.join(self._checkpoint_dir, filename)
        torch.save(state_dict, filename)
        logger.info("Checkpoint saved to '{}'.".format(filename))

        if clean_checkpoints:
            self._saved_checkpoints.append(filename)
            self.clean_checkpoints()

    def load_checkpoint(self, filename, model, optimizer=None, scheduler=None):
        logger = get_logger()
        logger.info("Loading checkpoint from '{}'.".format(filename))

        state_dict = torch.load(filename, map_location=torch.device("cpu"))

        # load metadata
        if "metadata" in state_dict:
            metadata = state_dict["metadata"]
        else:
            # Legacy: for old snapshots only
            metadata = {}
            if "epoch" in state_dict:
                metadata["epoch"] = state_dict["epoch"]
            if "iteration" in state_dict:
                metadata["total_steps"] = state_dict["iteration"]

        # load model
        if isinstance(model, DistributedDataParallel):
            model = model.module
        load_state_dict(model, state_dict["model"])
        logger.info("Model loaded.")

        # load optimizer
        if optimizer is not None:
            if "optimizer" not in state_dict:
                logger.warn(f"Optimizer checkpoint not found in '{filename}'.")
            else:
                optimizer.load_state_dict(state_dict["optimizer"])
                logger.info("Optimizer loaded.")

        # load scheduler
        if scheduler is not None:
            if "scheduler" not in state_dict:
                logger.warn(f"Scheduler checkpoint not found in '{filename}'.")
            else:
                scheduler.load_state_dict(state_dict["scheduler"])
                logger.info("Scheduler loaded.")

        # load checkpoint manager
        if "ckpt_manager" in state_dict:
            self.load_state_dict(state_dict["ckpt_manager"])
            logger.info("CheckpointManager loaded.")

        return metadata

    @master_only
    def clean_checkpoints(self):
        # save all models
        if self._save_latest_models <= 0:
            return

        # clean old checkpoints
        while len(self._saved_checkpoints) > self._save_latest_models:
            oldest_checkpoint = self._saved_checkpoints.pop(0)
            if osp.exists(oldest_checkpoint):
                os.remove(oldest_checkpoint)

    def add_metric(self, metric, largest=True):
        assert metric not in self._best_scores, f"Duplicate metric: {metric}"
        self._best_scores[metric] = {"score": None, "largest": largest}

    def _compare_metric(self, metric, score):
        best_score = self._best_scores[metric]["score"]
        if best_score is None:
            return True
        if self._best_scores[metric]["largest"]:
            return score > best_score
        else:
            return score < best_score

    @master_only
    def update_best_model(self, metric_dict, metadata, model):
        logger = get_logger()
        for metric in self._best_scores:
            if metric not in metric_dict:
                logger.warn(f"Metric '{metric}' not found (available: {metric_dict.keys()}).")
                continue
            cur_score = metric_dict[metric]
            if self._compare_metric(metric, cur_score):
                self._best_scores[metric]["score"] = cur_score
                self.save_checkpoint(f"best_{metric}.pth", metadata, model)
                logger.info(f"New best '{metric}': {cur_score:.3f}.")
