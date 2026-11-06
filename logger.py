from loguru import logger

logger.add("/tmp/bitty-rl/train.log", rotation="50 MB")
