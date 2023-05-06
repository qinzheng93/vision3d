import sys
import warnings

import loguru
from torch.utils.tensorboard import SummaryWriter

from .distributed import is_master


class Logger:
    """Advanced logger with stderr, log file and TensorBoard support.

    When DistributedDataParallel is enabled, only ERROR logs are activated for slave processes.
    """

    def __init__(self, log_file=None, event_dir=None):
        is_master_node = is_master()

        self._logger = loguru.logger
        self._logger.remove()
        fmt_str = "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level}</level> | <level><n>{message}</n></level>"
        log_level = "DEBUG" if is_master_node else "ERROR"
        self._logger.add(sys.stderr, format=fmt_str, colorize=True, level=log_level)
        self._logger.info("Command executed: " + " ".join(sys.argv))

        self._log_file = log_file if is_master_node else None
        self._event_dir = event_dir if is_master_node else None

        if self._log_file is not None:
            fmt_str = "{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
            self._logger.add(self._log_file, format=fmt_str, level="INFO")
            self._logger.info(f"Logs are saved to {self._log_file}.")

        if self._event_dir is not None:
            self._writer = SummaryWriter(log_dir=self._event_dir)
            self._logger.info(f"Tensorboard is enabled. Write events to {self._event_dir}.")
        else:
            self._writer = None

    @property
    def log_file(self):
        return self._log_file

    @property
    def event_dir(self):
        return self._event_dir

    def log(self, message, level="INFO"):
        if level not in ["DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]:
            self._logger.warning(f"Unsupported logging level: {level}. Fallback to INFO.")
            level = "INFO"
        self._logger.log(level, message)

    def debug(self, message):
        self._logger.debug(message)

    def info(self, message):
        self._logger.info(message)

    def success(self, message):
        self._logger.success(message)

    def warn(self, message):
        self._logger.warning(message)

    def error(self, message):
        self._logger.error(message)

    def critical(self, message):
        self._logger.critical(message)

    def write_event(self, phase, value, index):
        if self._writer is not None:
            self._writer.add_scalar(phase, value, index)


_LOGGER = None


def get_logger(log_file=None, event_dir=None):
    """Guarantee only one logger per node is built."""
    global _LOGGER
    if _LOGGER is None:
        _LOGGER = Logger(log_file=log_file, event_dir=event_dir)
    elif log_file is not None or event_dir is not None:
        log_strings = []
        if log_file is not None:
            log_strings.append(f"log_file={log_file}")
        if event_dir is not None:
            log_strings.append(f"event_dir={event_dir}")
        message = "Logger is already initialized. New parameters (" + ",".join(log_strings) + ") are ignored."
        warnings.warn(message)
    return _LOGGER
