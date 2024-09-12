import sys
import time
import socket
import logging

from typing import Optional, Callable


def set_logger(
    logger: logging.Logger,
    level: Optional[Callable] = None,
    stream: Optional[Callable] = sys.stdout,
    verbose: Optional[bool] = True,
) -> logging.Logger:
    """
    Set logging parameters,

    Parameters
    ----------
    logger: logging.Logger
        Logger object to set paramerts
    level: callable, optional, default 'logging.INFO'
        Print level for output (e.g. logging.DEBUG, logging.INFO, ...)
    stream: callable, optional, default 'sys.stdout'
        Output channel to print
    verbose: bool, optional, default True
        Start logger output with header for information

    Returns
    -------
    logging.Logger
        Loger object with set paramerts

    """

    # Set print level
    if level is None:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(level)

    handler = logging.StreamHandler(stream)
    if verbose:
        handler.setFormatter(
            logging.Formatter(
                '%(levelname)s - %(name)s.%(funcName)s:\n%(msg)s\n'))
    else:
        handler.setFormatter(logging.Formatter('%(msg)s\n'))

    logger.addHandler(handler)
    logger.propagate = False

    return logger


def get_header(
    config_file: str
) -> str:
    """
    Provide the Asparagus header.

    Parameters
    ----------
    config_file: str
        Current configuration file path

    Returns
    -------
    str
        Asparagus header

    """

    current_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    host = socket.gethostname()

    msg = (
        r"""        '   _______                                                  ______                  _ _
        '  (_______)                                                (____  \                | | |
        '   _______  ___ ____  _____  ____ _____  ____ _   _  ___    ____)  )_   _ ____   __| | | _____
        '  |  ___  |/___)  _ \(____ |/ ___|____ |/ _  | | | |/___)  |  __  (| | | |  _ \ / _  | || ___ |
        '  | |   | |___ | |_| / ___ | |   / ___ ( (_| | |_| |___ |  | |__)  ) |_| | | | ( (_| | || ____|
        '  |_|   |_(___/|  __/\_____|_|   \_____|\___ |____/(___/   |______/|____/|_| |_|\____|\_)_____)
        '               |_|                     (_____|"""
        + f"""
        '
        '                        Authors: K. Toepfer and L.I. Vazquez-Salazar
        '                        Date: {current_time:s}
        '                        Running on: {host:s}
        '                        Details of this run are stored in: {config_file:s}
        ' ---------------------------------------------------------------------------------------------"""
       )

    return msg


def print_ProgressBar(
    iteration: int,
    total: int,
    prefix: Optional[str] = '',
    suffix: Optional[str] = '',
    decimals: Optional[int] = 1,
    length: Optional[int] = 100,
    fill: Optional[str] = '#',
    printEnd: Optional[str] = '\r',
):
    """
    Call in a loop to create terminal progress bar

    Parameters
    ----------
    iteration: int
        current iteration (Int)
    total: int
        total iterations
    prefix: str, optional, default ''
        prefix string
    suffix: str, optional, default ''
        suffix string
    decimals: int, optional, default 1
        positive number of decimals in percent complete
    length: int, optional, default 100
        Character length of bar (Int)
    fill: str, optional, default '#'
        bar fill character
    printEnd: str, optional, default '\r'
        end character (e.g. "/r", "/r/n") (Str)

    """

    percent = (
        ("{0:." + str(decimals) + "f}").format(
            100 * (iteration / float(total)))
        )
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)

    print("\r{0} |{1}| {2}% {3}".format(
        prefix, bar, percent, suffix), end=printEnd)

    # Print New Line on Complete
    if iteration == total:
        print()

    return
