import logging
log_format = '%(asctime)s - %(levelname)s - %(message)s'


def setup_thread_logger(thread_name):
    # 创建一个独立的日志处理器，文件名根据线程名称
    file_handler = logging.FileHandler(f'{thread_name}.log')
    formatter = logging.Formatter(log_format)
    file_handler.setFormatter(formatter)
    # 创建线程独立的 logger
    thread_logger = logging.getLogger(thread_name)
    thread_logger.setLevel(logging.INFO)
    thread_logger.addHandler(file_handler)
    return thread_logger
