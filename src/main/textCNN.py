from tensorflow.keras import Input, regularizers, Model
from tensorflow.keras.layers import Embedding, Reshape, Conv2D, MaxPool2D
from tensorflow.keras.layers import concatenate, Flatten, Dropout, Dense
from tensorflow.keras.initializers import RandomUniform, constant

def TextCNN(vocab_size, feature_size, embed_size, num_classes, num_filters,
            filter_sizes, regularizes_lambda, dropout_rate):
    inputs = Input(shape=(feature_size,))
    # Ƕ���
    model = Embedding(input_dim=vocab_size, output_dim=embed_size,
                      embeddings_initializer=RandomUniform(minval=-1, maxval=1),
                      input_length=feature_size)(inputs)
    # ����ͨ��
    model = Reshape((feature_size, embed_size, 1))(model)

    pool_outputs = []
    for filter_size in list(map(int, filter_sizes.split(','))):
        # ���������СΪ3��4��5��ÿ������2�����˵�6������
        conv = Conv2D(num_filters, (filter_size, embed_size), strides=(1, 1),
                      padding='valid', data_format='channels_last', 
                      activation='relu', kernel_initializer='glorot_normal',
                      bias_initializer=constant(0.1),
                      name='Conv2D_{:d}'.format(filter_size))(model) 
        # ����ÿ������2������map�ĳػ���
        pool = MaxPool2D(pool_size=(feature_size - filter_size + 1, 1),
                         strides=(1, 1), padding='valid',
                         data_format='channels_last',
                         name='MaxPool2D_{:d}'.format(filter_size))(conv)
        pool_outputs.append(pool)

    # ƽ����
    pool_outputs = Flatten(data_format='channels_last')(
        concatenate(pool_outputs, axis=-1))
    # �ݶ��½���
    pool_outputs = Dropout(dropout_rate)(pool_outputs)

    # ȫ���Ӳ㣬������2ά�Ľ����
    outputs = Dense(num_classes, activation='softmax',
                    kernel_initializer='glorot_normal',
                    bias_initializer=constant(0.1),
                    kernel_regularizer=regularizers.l2(regularizes_lambda),
                    bias_regularizer=regularizers.l2(regularizes_lambda))(pool_outputs)
    return Model(inputs=inputs, outputs=outputs)