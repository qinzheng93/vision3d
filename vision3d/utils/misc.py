import importlib
from functools import wraps

import torch

# deprecate decorator


def deprecated(replacement: str):
    def decorate(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            raise RuntimeError(f"{func.__name__} is deprecated. Use {replacement} instead.")

        return wrapper

    return decorate


# importlib utilities


def load_ext(name, functions):
    ext_module = importlib.import_module(name)
    for function in functions:
        assert hasattr(ext_module, function), f"Function '{function}' missing in '{name}'."
    return ext_module


# log utilities


def get_print_format(value):
    if isinstance(value, (int, str)):
        return ""
    if value == 0:
        return ".3f"
    if value < 1e-5:
        return ".3e"
    if value < 1e-2:
        return ".6f"
    return ".3f"


def get_format_strings(result_dict):
    """Get format string for a list of key-value pairs."""
    format_strings = []
    if "metadata" in result_dict:
        # handle special key "metadata"
        format_strings.append(result_dict["metadata"])
    for key, value in result_dict.items():
        if key == "metadata":
            continue
        format_string = f"{key}: {value:{get_print_format(value)}}"
        format_strings.append(format_string)
    return format_strings


def get_log_string(
    result_dict, epoch=None, max_epoch=None, iteration=None, max_iteration=None, lr=None, time_dict=None
):
    log_strings = []
    if epoch is not None:
        epoch_string = f"epoch: {epoch}"
        if max_epoch is not None:
            epoch_string += f"/{max_epoch}"
        log_strings.append(epoch_string)
    if iteration is not None:
        iter_string = f"iter: {iteration}"
        if max_iteration is not None:
            iter_string += f"/{max_iteration}"
        log_strings.append(iter_string)
    log_strings += get_format_strings(result_dict)
    if lr is not None:
        log_strings.append("lr: {:.3e}".format(lr))
    if time_dict is not None:
        time_string = "time: " + "/".join([f"{time_dict[key]:.3f}s" for key in time_dict])
        log_strings.append(time_string)
    message = ", ".join(log_strings)
    return message
