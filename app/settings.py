import logging.config
from app import common

# FACE MODEL ENV VARIABLE
MODEL_PATH = "/opt/face/model_file/high_pclowloss.pth"
DETECTOR_MODEL_PATH = 'weights/pre_high_pc_rotatelowloss.pth'
CONFIDENCE_THRES = float(common.getEnv('CONFIDENCE_THRES', "0.9"))
SCALE_UP = float(common.getEnv("SCALE_UP", "0.0"))
INPUT_SIZE = float(common.getEnv("INPUT_SIZE", "1280."))
USE_GPU = common.getEnv("USE_GPU", "1") == "1"
LOGGING_LEVELS = {logging.INFO, logging.DEBUG, logging.WARNING, logging.ERROR, logging.CRITICAL}
LOGGING_LEVEL_NAMES = {logging.getLevelName(lvl) for lvl in LOGGING_LEVELS}
LOG_LEVEL_CONFIG = logging.DEBUG
# LOG_LEVEL_CONFIG = LOG_LEVEL_CONFIG if LOG_LEVEL_CONFIG in LOGGING_LEVEL_NAMES else logging.getLevelName(logging.INFO)

LOGGING_CONFIG = {
    "version": 1,
    "handlers":
        {
            "logfile": {
                "level": logging.INFO,
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "json-fmt",
                "filename": "/workspace/logs/facedt_service.log",
                "when": "midnight",
                "backupCount": 2
            },
            "stdout": {
                "level": logging.DEBUG,
                "class": "logging.StreamHandler",
                "formatter": "default"
            },
            "stdout-wrk": {
                "level": logging.WARNING,
                "class": "logging.StreamHandler",
                "formatter": "json-with-rid-fmt"
            }
        },
    "formatters": {
        "default": {
            "format": "%(levelname)s - %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S"
        },
        "json-fmt": {
            "format": '{"time":"%(asctime)s.%(msecs)03d", "level":"%(levelname)s", "msg":"%(message)s", "ServiceName":"' + 'FACEDT_SERVICE' + '", "funcName":"%(funcName)s", "process":"%(process)d", "thread":"%(thread)d"}',
            "datefmt": "%Y-%m-%dT%H:%M:%S"
        },
        "json-with-rid-fmt": {
            "format": '{"time":"%(asctime)s.%(msecs)03d", "level":"%(levelname)s", "msg":"%(message)s", "ServiceName":"' + 'FACEDT_SERVICE' + '", "funcName":"%(funcName)s", "process":"%(process)d", "thread":"%(thread)d", "redactionId": "%(redactionId)s", "fromRedaction": "%(fromRedaction)s"}',
            "datefmt": "%Y-%m-%dT%H:%M:%S"
        }
    },
    "loggers": {
        "default": {
            "level": LOG_LEVEL_CONFIG,
            "handlers": ['stdout', 'logfile']
        },
        "default-wrk": {
            "level": LOG_LEVEL_CONFIG,
            "handlers": ['stdout-wrk']
        }
    }
}
logging_config = logging.config.dictConfig(LOGGING_CONFIG)
logger = logging.getLogger("default")

BATCH_SIZE = 32
MATCHING = 1
MIN_SCORE = 0.9
DETECTION_COLOR = (0, 255, 0)
TRACK_COLOR = (0, 0, 255)
