# 鉴别器和生成器通用设置
EMB_DIM = 32 # 嵌入维度
HIDDEN_DIM = 32 # lstm单元的隐藏状态维度
SEQ_LENGTH = 20 # 序列长度（字数）
MIN_SEQ_LENGTH = 10 # 最小序列长度 
BATCH_SIZE = 64

# 判别器模型设置 

# 词的embedding选用的是64
dis_embedding_dim = 64 

# 定义CNN中的卷积核大小
dis_filter_sizes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20]

# 定义CNN中的卷积核数量
dis_num_filters = [100, 200, 200, 200, 200, 100, 100, 100, 100, 100, 160, 160]
#dropout是指在深度学习网络的训练过程中，对于神经网络单元，按照一定的概率将其暂时从网络中丢弃
dis_dropout_keep_prob = 0.75 #判别器神经网络丢弃保留率，用于防止过拟合
#GBDT (Gradient Boosting Decision Tree) 是机器学习中一个长盛不衰的模型，
#其主要思想是利用弱分类器（决策树）迭代训练以得到最优模型，该模型具有训练效果好、不易过拟合等优点。
#L2正则化参数，别名：lambda_l2。默认设置为0。
#较大的数值会让各个特征对模型的影响力趋于均匀，不会有单个特征把持整个模型的表现。需要调节来控制过拟合
dis_l2_reg_lambda = 0.2

LSTM_VERSION = 'lstm_2'

# Epoch Number
PRE_EPOCH_NUM = 120
EPOCH_NUM = 1
#批量生成文本数据
generated_num = 10000

vocab_size = 20000

# Dataset
dataset_path = 'dataset/IMDB Dataset.csv'
positive_file = 'dataset/positives.txt'
negative_file = 'dataset/negatives.txt'
generated_file = 'dataset/generated_file.txt'

# Saved Models
pretrained_generator_file = "pretrained_models/pretrained_generator.h5"
pretrained_discriminator_file = "pretrained_models/pretrained_discriminator.h5"
generator_file = "pretrained_models/generator.h5"
discriminator_file = "pretrained_models/discriminator.h5"

tokenizer_file = 'pretrained_models/tokenizer.pickle'
