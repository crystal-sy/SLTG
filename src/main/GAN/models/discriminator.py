from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense
import os
import sys
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

from self_attention import Self_Attention

model_dir = project_path + '\\result\\lstm_attention\\'

#Highway Networks 允许信息高速无阻碍的通过深层神经网络的各层，这样有效的减缓了梯度的问题，使深层神经网络不在仅仅具有浅层神经网络的效果。
class Highway(Model):
    """
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
     """
    def __init__(self, **kwargs):
        super(Highway, self).__init__(**kwargs)

    def build(self, input_shape):
        output_dim = input_shape[-1]
        self.dense_g = Dense(output_dim, activation="relu")
        self.dense_t = Dense(output_dim, activation="sigmoid")

    def call(self, input_tensor, training=False):
        g = self.dense_g(input_tensor, training=training)
        t = self.dense_t(input_tensor, training=training)
        o = t * g + (1. - t) * input_tensor
        return o
    
class Discriminator:
    def __init__(self, version):
        self.version = version

    def train(self, dataset, num_epochs, num_steps, **kwargs):
        return self.d_model.fit(dataset.repeat(num_epochs), verbose=1, epochs=num_epochs, steps_per_epoch=num_steps,
                                **kwargs)

    def save(self, filename):
        self.d_model.save(filename)

    def load(self):
        self.d_model = load_model(model_dir + self.version + '/lstm.h5', 
                                 custom_objects = {'Self_Attention': Self_Attention})
        # self.d_model.add(Highway())