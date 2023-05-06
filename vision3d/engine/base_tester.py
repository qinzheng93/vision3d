import abc
import json
import os.path as osp
import time

import torch

from vision3d.utils.logger import get_logger
from vision3d.utils.misc import get_log_string
from vision3d.utils.parser import get_default_parser

from .checkpoint import load_state_dict
from .utils import setup_engine


class BaseTester(abc.ABC):
    def __init__(self, cfg):
        # parser
        parser = get_default_parser()
        self._args = parser.parse_args()
        self._cudnn_deterministic = self._args.cudnn_deterministic

        # cuda check
        assert torch.cuda.is_available(), "No CUDA devices available."

        # logger
        self._log_file = osp.join(cfg.exp.log_dir, "test-{}.log".format(time.strftime("%Y%m%d-%H%M%S")))
        self._logger = get_logger(log_file=self._log_file)

        # find checkpoint
        self._checkpoint = self._args.checkpoint
        if self._checkpoint is None and self._args.test_epoch is not None:
            self._checkpoint = f"epoch-{self._args.test_epoch}.pth"
        if self._checkpoint is not None:
            self._checkpoint = osp.join(cfg.exp.checkpoint_dir, self._checkpoint)

        # print config
        self.log("Configs:\n" + json.dumps(cfg, indent=4))

        # initialize
        setup_engine(seed=cfg.exp.seed, cudnn_deterministic=self._cudnn_deterministic)

        # state
        self.model = None
        self.iteration = None

        # data loader
        self.test_loader = None

    @property
    def args(self):
        return self._args

    @property
    def log_file(self):
        return self._log_file

    def load(self, filename, strict=True):
        self.log('Loading from "{}".'.format(filename))
        state_dict = torch.load(filename, map_location=torch.device("cpu"))
        assert "model" in state_dict, "No model can be loaded."
        load_state_dict(self.model, state_dict["model"], strict=strict)
        self.log("Model has been loaded.")
        if "metadata" in state_dict:
            epoch = state_dict["metadata"]["epoch"]
            total_steps = state_dict["metadata"]["total_steps"]
            self.log(f"Checkpoint metadata: epoch: {epoch}, total_steps: {total_steps}.")

    def register_model(self, model):
        """Register model."""
        model = model.cuda()
        self.model = model
        message = "Model description:\n" + str(model)
        self.log(message)
        return model

    def register_loader(self, test_loader):
        """Register data loader."""
        self.test_loader = test_loader

    def log(self, message, level="INFO"):
        self._logger.log(message, level=level)

    def before_test_epoch(self):
        pass

    def before_test_step(self, iteration, data_dict):
        pass

    @abc.abstractmethod
    def test_step(self, iteration, data_dict) -> dict:
        pass

    @abc.abstractmethod
    def eval_step(self, iteration, data_dict, output_dict) -> dict:
        pass

    def after_test_step(self, iteration, data_dict, output_dict, result_dict):
        pass

    def after_test_epoch(self, summary_dict):
        pass

    def get_log_string(self, iteration, data_dict, output_dict, result_dict) -> str:
        return get_log_string(result_dict)

    @abc.abstractmethod
    def test_epoch(self):
        pass

    def run(self, strict_loading=True):
        assert self.test_loader is not None
        if self._checkpoint is not None:
            self.load(self._checkpoint, strict=strict_loading)
        else:
            self.log("Checkpoint is not specified.", level="WARNING")
        self.model.eval()
        torch.set_grad_enabled(False)
        self.test_epoch()
