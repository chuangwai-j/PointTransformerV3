# pointcept/utils/logger.py
import logging

def get_root_logger(log_file=None, log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.handlers:
        # 如果没有处理器，则添加一个
        format_str = '%(asctime)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_str)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)
        logger.setLevel(log_level)
        if log_file is not None:
            file_handler = logging.FileHandler(log_file, mode='a')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    return logger