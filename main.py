# main.py

import os
import toml
import argparse
from pprint import pprint

import torch
from torch.utils.data import DataLoader

import utils
from utils import CONFIG
from trainer import Trainer
from dataloader.data_generator import DataGenerator

def main():

    # Train or Test
    if CONFIG.phase.lower() == "train":

        # Создание директорий, если они не существуют.
        utils.make_dir(CONFIG.log.logging_path)
        utils.make_dir(CONFIG.log.tensorboard_path)
        utils.make_dir(CONFIG.log.checkpoint_path)

        # Создание логгеров
        logger, tb_logger = utils.get_logger(CONFIG.log.logging_path,
                                             CONFIG.log.tensorboard_path,
                                             logging_level=CONFIG.log.logging_level)

        train_dataset = DataGenerator(phase='train')

        train_sampler = None

        train_dataloader = DataLoader(train_dataset,
                                      batch_size=CONFIG.model.batch_size,
                                      shuffle=(train_sampler is None),
                                      num_workers=CONFIG.data.workers,
                                      pin_memory=True,
                                      sampler=train_sampler,
                                      drop_last=True)

        trainer = Trainer(train_dataloader=train_dataloader,
                          test_dataloader=None,
                          logger=logger,
                          tb_logger=tb_logger)

        trainer.train()
    else:
        raise NotImplementedError(f"Unknown Phase: {CONFIG.phase}")

if __name__ == '__main__':
    print('Torch Version: ', torch.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='train')
    parser.add_argument('--config', type=str, default='config/gca-dist.toml')
    # Parse configuration
    args = parser.parse_args()
    with open(args.config) as f:
        utils.load_config(toml.load(f))

    # Check if toml config file is loaded
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.")

    CONFIG.phase = args.phase
    CONFIG.log.logging_path = os.path.join(CONFIG.log.logging_path, CONFIG.version)
    CONFIG.log.tensorboard_path = os.path.join(CONFIG.log.tensorboard_path, CONFIG.version)
    CONFIG.log.checkpoint_path = os.path.join(CONFIG.log.checkpoint_path, CONFIG.version)
    
    if True:
        print('CONFIG: ')
        pprint(CONFIG)
    
    # Установите local_rank в 0, если не используете распределённое обучение
    CONFIG.local_rank = 0

    # Train
    main()
