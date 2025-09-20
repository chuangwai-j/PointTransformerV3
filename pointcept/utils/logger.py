"""
Logger Utils

Modified from mmcv

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

import logging
import torch
import torch.distributed as dist
import os  # 新增：导入os模块处理路径

from termcolor import colored

logger_initialized = {}
root_status = 0


class _ColorfulFormatter(logging.Formatter):
    def __init__(self, *args, **kwargs):
        self._root_name = kwargs.pop("root_name") + "."
        super(_ColorfulFormatter, self).__init__(*args, **kwargs)

    def formatMessage(self, record):
        log = super(_ColorfulFormatter, self).formatMessage(record)
        if record.levelno == logging.WARNING:
            prefix = colored("WARNING", "red", attrs=["blink"])
        elif record.levelno == logging.ERROR or record.levelno == logging.CRITICAL:
            prefix = colored("ERROR", "red", attrs=["blink", "underline"])
        else:
            return log
        return prefix + " " + log


def get_logger(name, log_file=None, log_dir=None, log_level=logging.INFO, file_mode="a", color=False):
    """Initialize and get a logger by name.

    新增参数:
        log_dir (str | None): 日志文件保存目录。若指定，将自动生成日志文件名（{name}.log），
            优先级低于log_file（若同时指定log_file，以log_file为准）。

    其他参数说明不变...
    """
    logger = logging.getLogger(name)

    if name in logger_initialized:
        return logger
    # 处理层级名称（保持原有逻辑）
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    logger.propagate = False

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # 新增：处理log_dir参数，自动生成log_file路径
    if log_dir is not None and log_file is None:
        os.makedirs(log_dir, exist_ok=True)  # 确保目录存在
        log_file = os.path.join(log_dir, f"{name}.log")  # 生成文件名：{name}.log

    # 仅rank=0添加FileHandler（保持原有逻辑）
    if rank == 0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    # 格式化器（保持原有逻辑）
    plain_formatter = logging.Formatter(
        "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d] %(message)s"
    )
    if color:
        formatter = _ColorfulFormatter(
            colored("[%(asctime)s %(name)s]: ", "green") + "%(message)s",
            datefmt="%m/%d %H:%M:%S",
            root_name=name,
        )
    else:
        formatter = plain_formatter
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    # 保持原有逻辑不变
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == "silent":
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            "logger should be either a logging.Logger object, str, "
            f'"silent" or None, but got {type(logger)}'
        )


def get_root_logger(log_file=None, log_dir=None, log_level=logging.INFO, file_mode="a"):
    """Get the root logger.

    新增参数:
        log_dir (str | None): 根日志保存目录（同get_logger逻辑）
    """
    logger = get_logger(
        name="Pointcept",
        log_file=log_file,
        log_dir=log_dir,  # 传递log_dir参数
        log_level=log_level,
        file_mode=file_mode
    )
    return logger


def _log_api_usage(identifier: str):
    # 保持原有逻辑不变
    torch._C._log_api_usage_once("Pointcept." + identifier)
