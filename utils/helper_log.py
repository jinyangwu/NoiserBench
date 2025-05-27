#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import logging


def create_log_id(dir_path):
    '''Create the log_count file'''
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def set_up_logger(log_path, level=logging.INFO, console_level=logging.INFO, file_handler=False, stream_handler=False):    
    '''Setup logger to print some information while running which is more convenient than print'''
    log_save_id = create_log_id(log_path)  # get id of log file and then generate a name
    name = f'log{log_save_id:d}'

    logger = logging.root
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    for handler in logging.root.handlers:
        logger.removeHandler(handler)
    logger.handlers = []
    path = os.path.join(log_path, name + ".log")
    logger.setLevel(level)    
    formatter = logging.Formatter(     
        fmt='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d,%H:%M:%S'
    )
    print("All logs will be saved to %s" % path)

    if file_handler:    # output the log to the log_file 'xxx.log'
        logfile = logging.FileHandler(path)
        logfile.setLevel(level)
        logfile.setFormatter(formatter)
        logger.addHandler(logfile)

    if stream_handler:  # output the log to the screen
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logger.addHandler(logconsole)
    return logger