import abc
from typing import Tuple, List, Dict

import torch
from tqdm import tqdm

from vision3d.utils.misc import get_log_string
from vision3d.utils.summary_board import SummaryBoard
from vision3d.utils.tensor import move_to_cuda, tensor_to_array
from vision3d.utils.timer import Timer

from .base_tester import BaseTester
from .context_manager import clear_context_manager


class BatchTester(BaseTester, abc.ABC):
    def __init__(self, cfg):
        super().__init__(cfg)

    @abc.abstractmethod
    def split_batch_dict(self, iteration, data_dict, output_dict) -> Tuple[List[Dict], List[Dict]]:
        pass

    def test_epoch(self):
        # before epoch
        self.before_test_epoch()
        # initialize watcher
        summary_board = SummaryBoard(auto_register=True)
        timer = Timer()
        # test loop
        pbar = tqdm(total=len(self.test_loader.dataset))
        step_index = 0
        timer.tic("data")
        for batch_index, data_dict in enumerate(self.test_loader):
            clear_context_manager()
            # load data
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
            # sample-wise eval step
            timer.tic("data")
            batch_data_dict, batch_output_dict = self.split_batch_dict(self.iteration, data_dict, output_dict)
            for single_data_dict, single_output_dict in zip(batch_data_dict, batch_output_dict):
                step_index += 1
                pbar.update()
                # eval step
                single_result_dict = self.eval_step(step_index, data_dict, output_dict)
                single_result_dict = tensor_to_array(single_result_dict)
                # after step
                self.after_test_step(step_index, single_data_dict, single_output_dict, single_result_dict)
                # logging
                single_result_dict = tensor_to_array(single_result_dict)
                summary_board.update_from_dict(single_result_dict)
                message = self.get_log_string(step_index, single_data_dict, single_output_dict, single_result_dict)
                pbar.set_description(message + ", " + timer.tostring(keys=["data", "model"], verbose=False))
            torch.cuda.empty_cache()
        pbar.close()
        # summary logging
        summary_dict = summary_board.summary()
        message = get_log_string(summary_dict, time_dict=timer.summary(["data", "model"]))
        self.log(message, level="SUCCESS")
        # after epoch
        self.after_test_epoch(summary_dict)
