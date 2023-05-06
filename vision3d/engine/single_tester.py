import abc

import torch
from tqdm import tqdm

from vision3d.utils.misc import get_log_string
from vision3d.utils.summary_board import SummaryBoard
from vision3d.utils.tensor import move_to_cuda, tensor_to_array
from vision3d.utils.timer import Timer

from .base_tester import BaseTester
from .context_manager import clear_context_manager


class SingleTester(BaseTester, abc.ABC):
    def __init__(self, cfg):
        super().__init__(cfg)

    def test_epoch(self):
        # before epoch
        self.before_test_epoch()
        # setup watcher
        summary_board = SummaryBoard(auto_register=True)
        timer = Timer()
        # test loop
        pbar = tqdm(enumerate(self.test_loader), total=len(self.test_loader))
        timer.tic("data")
        for batch_index, data_dict in pbar:
            clear_context_manager()
            # on start
            self.iteration = batch_index + 1
            data_dict = move_to_cuda(data_dict)
            self.before_test_step(self.iteration, data_dict)
            timer.toc("data")
            # test step
            torch.cuda.synchronize()
            timer.tic("model")
            output_dict = self.test_step(self.iteration, data_dict)
            torch.cuda.synchronize()
            timer.toc("model")
            # eval step
            timer.tic("data")
            result_dict = self.eval_step(self.iteration, data_dict, output_dict)
            # after step
            self.after_test_step(self.iteration, data_dict, output_dict, result_dict)
            # logging
            result_dict = tensor_to_array(result_dict)
            summary_board.update_from_dict(result_dict)
            message = self.get_log_string(self.iteration, data_dict, output_dict, result_dict)
            pbar.set_description(message + ", " + timer.tostring(keys=["data", "model"], verbose=False))
            torch.cuda.empty_cache()
        # summary logging
        summary_dict = summary_board.summary()
        message = get_log_string(summary_dict, time_dict=timer.summary(keys=["data", "model"]))
        self.log(message, level="SUCCESS")
        # after epoch
        self.after_test_epoch(summary_dict)
