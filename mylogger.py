from logging import INFO, Formatter, getLogger, handlers


def set_logger():
    root_logger = getLogger()
    root_logger.setLevel(INFO)
    rotating_handler = handlers.RotatingFileHandler(
        r"./output.log", mode="a", maxBytes=100 * 1024, backupCount=3, encoding="utf-8"
    )
    format = Formatter("%(asctime)s : %(levelname)s : %(filename)s - %(message)s")
    rotating_handler.setFormatter(format)
    root_logger.addHandler(rotating_handler)
