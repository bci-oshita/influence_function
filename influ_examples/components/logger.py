from logging import Logger

g_logger = Logger()


def log(*args, **kwargs):
    g_logger.info(*args, **kwargs)
