import logging
import os

def getLogger(name:str, filename="predict_classification"):
    """Create a logger with the given name and filename."""
    
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    os.makedirs('./logs', exist_ok=True)
    fh = logging.FileHandler(filename='./logs/' + filename + '.log')
    fh.setLevel(logging.DEBUG)

    fmt = '%(asctime)s %(name)-12s %(levelname)-8s %(message)s'  #'[%(asctime)s] [%(process)d] [%(levelname)s] [%(name)s] %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S %z'
    formatter = logging.Formatter(fmt, datefmt)

    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    return logger