from tensorflow.keras.models import load_model
from tensorflow.keras import backend
from tensorflow.keras.layers import Layer

model_dir = 'D:\\sycode\\SLTG\\src\\main\\result\\lstm_attention\\'

class Self_Attention(Layer):
    def __init__(self, output_dim, **kwargs):
        self.output_dim = output_dim
        super(Self_Attention, self).__init__(**kwargs)
        
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'output_dim': self.output_dim 
        })
        return config

    def build(self, input_shape):
        # 为该层创建一个可训练的权重
        #inputs.shape = (batch_size, time_steps, seq_len)
        self.kernel = self.add_weight(name='kernel',
                                      shape=(3,input_shape[2], self.output_dim),
                                      initializer='uniform',
                                      trainable=True)
        super(Self_Attention, self).build(input_shape)  # 一定要在最后调用它

    def call(self, x):
        WQ = backend.dot(x, self.kernel[0])
        WK = backend.dot(x, self.kernel[1])
        WV = backend.dot(x, self.kernel[2])

        print("WQ.shape",WQ.shape)
        print("K.permute_dimensions(WK, [0, 2, 1]).shape",backend.permute_dimensions(WK, [0, 2, 1]).shape)

        QK = backend.batch_dot(WQ, backend.permute_dimensions(WK, [0, 2, 1]))
        QK = QK / (64**0.5) #开根号，归一化系数
        QK = backend.softmax(QK)
        print("QK.shape", QK.shape)

        V = backend.batch_dot(QK,WV)
        return V

    def compute_output_shape(self, input_shape):
        return (input_shape[0],input_shape[1],self.output_dim)

class Discriminator:
    def __init__(self, version):
        self.version = version

    def train(self, dataset, num_epochs, num_steps, **kwargs):
        return self.d_model.fit(dataset.repeat(num_epochs), verbose=1, epochs=num_epochs, steps_per_epoch=num_steps,
                                **kwargs)

    def save(self, filename):
        self.d_model.save(filename)

    def load(self):
        self.d_model= load_model(model_dir + self.version + '/lstm.h5', 
                                 custom_objects = {'Self_Attention': Self_Attention})
