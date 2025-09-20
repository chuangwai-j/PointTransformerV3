# pointcept/utils/logging.py
import logging
import os
from datetime import datetime


def setup_logging(log_dir="logs", console_level=logging.WARNING):
    """
    配置全局日志：调试信息写文件，终端只显警告及以上
    :param log_dir: 日志文件目录
    :param console_level: 终端输出级别（默认WARNING，只显警告/错误）
    :return: 配置好的根日志器
    """
    # 1. 获取根日志器（全局唯一）
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)  # 根日志器捕获所有级别（debug及以上）

    # 2. 清除已有处理器（避免重复配置导致日志重复输出）
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    # 3. 创建日志目录
    os.makedirs(log_dir, exist_ok=True)

    # 4. 配置「文件处理器」：所有调试信息写入文件（含时间戳）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"train_{timestamp}.log")
    file_handler = logging.FileHandler(log_file, encoding="utf-8")  # 加encoding避免中文乱码
    file_handler.setLevel(logging.DEBUG)  # 文件记录所有调试信息
    file_formatter = logging.Formatter(
        "%(asctime)s - %(module)s:%(lineno)d - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )  # 日志格式：时间+文件名+行号+级别+内容（方便定位问题）
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # 5. 配置「控制台处理器」：只输出警告及以上（不干扰进度条）
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_formatter = logging.Formatter("%(levelname)s: %(message)s")  # 控制台格式简化
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    return root_logger