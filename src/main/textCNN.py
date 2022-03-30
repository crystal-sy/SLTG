from tensorflow.keras import Input, regularizers, Model
from tensorflow.keras.layers import Embedding, Reshape, Conv2D, MaxPool2D
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense
from tensorflow.keras.initializers import RandomUniform, constant

def TextCNN(vocab_size, feature_size, embed_size, num_classes, num_filters,
            filter_sizes, regularizes_lambda, dropout_rate):
    inputs = Input(shape=(feature_size,))
    # 嵌入层
    model = Embedding(input_dim=vocab_size, output_dim=embed_size,
                      embeddings_initializer=RandomUniform(minval=-1, maxval=1),
                      input_length=feature_size)(inputs)
    # 增加通道
    model = Reshape((feature_size, embed_size, 1))(model)

    pool_outputs = []
    for filter_size in list(map(int, filter_sizes.split(','))):
        # 构造区域大小为3，4，5，每个区域2个过滤的6层卷积层
        conv = Conv2D(num_filters, (filter_size, embed_size), strides=(1, 1),
                      padding='valid', data_format='channels_last', 
                      activation='relu', kernel_initializer='glorot_normal',
                      bias_initializer=constant(0.1),
                      name='Conv2D_{:d}'.format(filter_size))(model) 
        # 构造每个区域2个特征map的池化层
        pool = MaxPool2D(pool_size=(feature_size - filter_size + 1, 1),
                         strides=(1, 1), padding='valid',
                         data_format='channels_last',
                         name='MaxPool2D_{:d}'.format(filter_size))(conv)
        pool_outputs.append(pool)

    # 平滑层
    pool_outputs = Flatten(data_format='channels_last')(
        concatenate(pool_outputs, axis=-1))
    # 梯度下降层
    pool_outputs = Dropout(dropout_rate)(pool_outputs)

    # 全连接层，最后输出2维的结果集
    outputs = Dense(num_classes, activation='softmax',
                    kernel_initializer='glorot_normal',
                    bias_initializer=constant(0.1),
                    kernel_regularizer=regularizers.l2(regularizes_lambda),
                    bias_regularizer=regularizers.l2(regularizes_lambda))(pool_outputs)
    return Model(inputs=inputs, outputs=outputs)