# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 18:19:10 2021

@author: styra
"""
import os
from utils.dataloader import generator_dataloader, discriminator_dataloader
from models.generator import Generator
from models.discriminator import Discriminator
from models.rollout import ROLLOUT
import gan_config as config
from config import sltg_config as sl_config
import sys
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

import logging
import logging.config
import warnings

warnings.filterwarnings('ignore')
    
logging.config.fileConfig(sl_config.logging_path)
logger = logging.getLogger('spider')

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

if __name__ == "__main__":
    generator = Generator(config.BATCH_SIZE, config.EMB_DIM,
                          config.HIDDEN_DIM, config.SEQ_LENGTH)
    discriminator = Discriminator(config.LSTM_VERSION)

    gen_dataset = generator_dataloader(config.real_file, config.BATCH_SIZE, config.SEQ_LENGTH)
    num_steps = config.generated_num // config.BATCH_SIZE

    if not os.path.exists("pretrained_models"):
        os.makedirs("pretrained_models")

    if not os.path.exists(config.pretrained_generator_file):
        logger.info(u'Start pre-training generator')
        generator.pretrain(gen_dataset, config.PRE_EPOCH_NUM, num_steps)
        generator.save(config.pretrained_generator_file)
        logger.info(u'Finished pre-training generator...')
    else:
        generator.load(config.pretrained_generator_file)

    discriminator.load()

    rollout = ROLLOUT(generator, 0.8)

    logger.info(u'==================================================================')
    logger.info(u'Start Adversarial Training...')
    # 那么我们要开始进行训练了。 规则： 训练生成器一次； 训练辨别器五次。
    for epoch in range(config.EPOCH_NUM):
        logger.info(u'Generator: %s', epoch)
        for it in range(1):
            samples = generator.generate_one_batch()
            logger.info(u'samples: %s', epoch)
            #基于生成器生成的数据和判别器计算rewards。
            rewards = rollout.get_reward(samples, 8, discriminator)
            logger.info(u'rewards: %s', epoch)
            generator.train_step(samples, rewards)
            logger.info(u'train_step: %s', epoch)

        # 用模型参数进行更新rollout。
        rollout.update_params()

        logger.info(u'Discriminator: %s', epoch)
        for i in range(5):
            # 根据训练的生成器模型，生成句子。
            generator.generate_samples(num_steps, config.fake_file)
            logger.info(u'generate_samples: %s', i)
            
            # 根据训练的生成器模型，生成句子。
            dis_dataset = discriminator_dataloader(config.real_file,
                                                   config.fake_file, 
                                                   config.BATCH_SIZE,
                                                   config.SEQ_LENGTH)
            logger.info(u'dis_dataset: %s', i)
            discriminator.train(dis_dataset, 1, num_steps * 2)
    generator.save(config.generator_file)
    discriminator.save(config.discriminator_file)

    generator.generate_samples(num_steps, config.generated_file)
