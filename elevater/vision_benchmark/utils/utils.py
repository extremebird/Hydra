from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from pathlib import Path

import os
import logging
import time

from .comm import comm


def setup_logger(final_output_dir, rank, phase):
    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = f'{phase}_{time_str}_rank{rank}.txt'
    final_log_file = os.path.join(final_output_dir, log_file)
    head = "%(asctime)-15s:[P:%(process)d]:" + comm.head + ' %(message)s'

    logger = logging.getLogger('my_logger')
    logger.propagate=False
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(final_log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(head))
    logger.addHandler(file_handler)

    console = logging.StreamHandler()
    console.setFormatter(logging.Formatter(head))
    logger.addHandler(console)


def create_logger(cfg, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    dataset = cfg.DATASET.DATASET
    cfg_name = cfg.NAME

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {} ...'.format(root_output_dir))
    root_output_dir.mkdir(parents=True, exist_ok=True)
    print('=> creating {} ...'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    print('=> setup logger ...')
    setup_logger(final_output_dir, cfg.RANK, phase)

    return str(final_output_dir)

