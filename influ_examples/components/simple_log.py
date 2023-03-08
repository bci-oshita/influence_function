import logging
import sys


log_format = "%(asctime)s.%(msecs)03d %(name)s %(levelname)s %(filename)s:%(lineno)s %(message)s"
logging.basicConfig(stream=sys.stdout, format=log_format, level=logging.INFO)
# logging.basicConfig(filename="train.log", filemode="w", format=log_format, level=logging.INFO)
g_logger = logging.getLogger(__name__)


def log(*args, **kwargs):
    logging.info(*args, **kwargs)
