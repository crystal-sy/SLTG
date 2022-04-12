from models.rnn import RNN

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

import os
import sys
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

from config import sltg_config as sl_config
import logging
import logging.config
import warnings

warnings.filterwarnings('ignore')
    
logging.config.fileConfig(sl_config.logging_path)
logger = logging.getLogger('spider')


class Generator(RNN):
    def __init__(self, batch_size, emb_dim, hidden_dim, sequence_length, learning_rate=0.01):
        super(Generator, self).__init__(batch_size, emb_dim, hidden_dim, sequence_length, learning_rate)
        #容器构建神经网络，依据层名或下标获得层对象
        self.generator_model = Sequential(self.generator_model.layers)
        self.generator_optimizer = self.create_optimizer(
            learning_rate,
            clipnorm=self.grad_clip
        )
        self.generator_model.compile(
            optimizer=self.generator_optimizer,
            loss="sparse_categorical_crossentropy",
            sample_weight_mode="temporal")

    def pretrain(self, dataset, num_epochs, num_steps):
        ds = dataset.map(lambda x: (tf.pad(x[:, 0:-1], ([0, 0], [1, 0]), "CONSTANT", 0), x)).repeat(
            num_epochs)
        pretrain_loss = self.generator_model.fit(ds, verbose=1, epochs=num_epochs, steps_per_epoch=num_steps)
        logger.info(u'Pretrain generator loss: : %s', pretrain_loss)
        return pretrain_loss

    def train_step(self, x, rewards):
        train_loss = self.generator_model.train_on_batch(
            np.pad(x[:, 0:-1], ([0, 0], [1, 0]), "constant", constant_values=0),
            x,
            sample_weight=rewards * self.batch_size * self.sequence_length
        )
        logger.info(u'Generator Loss: : %s', train_loss)
        return train_loss

    def create_optimizer(self, *args, **kwargs):
        #实现自适应估计的随机梯度下降方法的优化器
        return Adam(*args, **kwargs)

    def save(self, filename):
        self.generator_model.save_weights(filename, save_format="h5")

    def load(self, filename):
        self.generator_model.load_weights(filename)
