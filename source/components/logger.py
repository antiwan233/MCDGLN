import logging
from pathlib import Path
from typing import Tuple
from omegaconf import DictConfig


# 设置logger的格式，返回一个logging.Formatter类型的参数
def get_formatter() -> logging.Formatter:
    return logging.Formatter('[%(asctime)s][%(filename)s][L%(lineno)d][%(levelname)s] %(message)s')


def initialize_logger() -> logging.Logger:

    # 实例化一个logger对象
    logger = logging.getLogger()

    # 设置logger的level为INFO
    logger.setLevel(logging.INFO)

    # handler.close() 回收处理器的系统资源
    for handler in logger.handlers:
        handler.close()

    # 清空logger.handlers列表
    logger.handlers.clear()

    # 设置formatter格式器
    formatter = get_formatter()

    # 实例化一个StreamHandler对象
    stream_handler = logging.StreamHandler()

    # 为handler设置formatter
    stream_handler.setFormatter(formatter)

    # 将handler添加到logger中
    logger.addHandler(stream_handler)

    return logger


# 设置logger的FileHandler
def set_file_handler(log_file_path: Path) -> logging.Logger:
    logger = initialize_logger()
    formatter = get_formatter()

    # 实例化一个FileHandler对象
    file_handler = logging.FileHandler(str(log_file_path))

    # 为handler设置formatter
    file_handler.setFormatter(formatter)

    # 将handler添加到logger中
    logger.addHandler(file_handler)

    return logger


def logger_factory() -> logging.Logger:

    # 生成一个log文件夹，用于存放log文件
    # 路径的名称为 result/unique_id
    # unique_id是和每一次训练/wandb run绑定的
    # log_path = Path(config.log_path) / config.unique_id
    # log_path.mkdir(exist_ok=True, parents=True)
    # logger = set_file_handler(log_file_path=log_path
    #                           / config.unique_id)

    logger = initialize_logger()

    return logger
