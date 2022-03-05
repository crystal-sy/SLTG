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
from gan_config import * 

if __name__ == "__main__":
    generator = Generator(vocab_size, BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH)
    discriminator = Discriminator(LSTM_VERSION)

    gen_dataset = generator_dataloader(positive_file, BATCH_SIZE)

    if not os.path.exists("pretrained_models"):
        os.makedirs("pretrained_models")

    if not os.path.exists(pretrained_generator_file):
        print('Start pre-training generator')
        generator.pretrain(gen_dataset, PRE_EPOCH_NUM, generated_num // BATCH_SIZE)
        generator.save(pretrained_generator_file)
        print('Finished pre-training generator...')
    else:
        generator.load(pretrained_generator_file)

    discriminator.load()

    rollout = ROLLOUT(generator, 0.8)

    print('=========================================================================')
    print('Start Adversarial Training...')
    # 那么我们要开始进行训练了。 规则： 训练生成器一次； 训练辨别器五次。
    for epoch in range(EPOCH_NUM):
        print("Generator", epoch)
        for it in range(1):
            samples = generator.generate_one_batch()
            #基于生成器生成的数据和判别器计算rewards。
            rewards = rollout.get_reward(samples, 16, discriminator)
            generator.train_step(samples, rewards)

        # 用模型参数进行更新rollout。
        rollout.update_params()

        print("Discriminator", epoch)
        for _ in range(5):
            
            # 根据训练的生成器模型，生成句子。
            generator.generate_samples(generated_num // BATCH_SIZE, negative_file)
            
            # 根据训练的生成器模型，生成句子。
            dis_dataset = discriminator_dataloader(positive_file, negative_file, BATCH_SIZE)
            discriminator.train(dis_dataset, 3, (generated_num // BATCH_SIZE) * 2)
    generator.save(generator_file)
    discriminator.save(discriminator_file)

    generator.generate_samples(generated_num // BATCH_SIZE, generated_file)
