import argparse
import os
from textCNN import TextCNN
from tensorflow import keras
from textCNN_data_helper import preprocess
import tensorflow as tf
from pprint import pprint
import time

# 加载整个模型结构
from tensorflow.keras.models import load_model
# 自然语言处理NLP神器--gensim，词向量Word2Vec
from matplotlib import pyplot as plt
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

def train(x_train, y_train, vocab_size, feature_size, save_path):
    print("\nTrain...")
    model = TextCNN(vocab_size, feature_size, args.embed_size, args.num_classes,
                    args.num_filters, args.filter_sizes, args.regularizes_lambda, args.dropout_rate)
    model.summary()
    # parallel_model = keras.utils.multi_gpu_model(model, gpus=2)
    #model.compile()方法用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
    model.compile(tf.optimizers.Adam(), loss='categorical_crossentropy',
                           metrics=['accuracy'])
    keras.utils.plot_model(model, show_shapes=True, to_file=os.path.join(args.results_dir, timestamp, "model.pdf"))
    y_train = tf.one_hot(y_train, args.num_classes)
    tb_callback = keras.callbacks.TensorBoard(os.path.join(args.results_dir, timestamp, 'log/'),
                                              histogram_freq=0.1, write_graph=True,
                                              write_grads=True, write_images=True,
                                              embeddings_freq=0.5, update_freq='batch')
    h = model.fit(x=x_train, y=y_train, batch_size=args.batch_size, epochs=args.epochs,
                                 callbacks=[tb_callback], validation_split=args.fraction_validation, shuffle=True)
    print("\nSaving model...")
    plt.plot(h.history["loss"],label="train_loss")
    plt.plot(h.history["val_loss"],label="val_loss")
    plt.plot(h.history["accuracy"],label="train_acc")
    plt.plot(h.history["val_accuracy"],label="val_acc")
    plt.legend()
    plt.savefig(os.path.join(save_path, 'result.png')) # show之前保存图片，之后保存图片为空白
    plt.show()
    
    keras.models.save_model(model, os.path.join(save_path, 'TextCNN.h5'))
    pprint(h.history)
    
    # 展开模型参数
    loadModel = load_model(os.path.join(save_path, 'TextCNN.h5'))
    with open(os.path.join(save_path, 'modelsummary.txt'), 'w') as f:
        loadModel.summary(print_fn=lambda x: f.write(x + '\n'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='This is the TextCNN train project.')
    parser.add_argument('-p', '--padding_size', default=128, type=int, help='Padding size of sentences.(default=128)')
    parser.add_argument('-e', '--embed_size', default=128, type=int, help='Word embedding size.(default=128)')
    parser.add_argument('-f', '--filter_sizes', default='3,4,5', help='Convolution kernel sizes.(default=3,4,5)')
    parser.add_argument('-n', '--num_filters', default=128, type=int, help='Number of each convolution kernel.(default=128)')
    parser.add_argument('-d', '--dropout_rate', default=0.5, type=float, help='Dropout rate in softmax layer.(default=0.5)')
    parser.add_argument('-c', '--num_classes', default=2, type=int, help='Number of target classes.(default=2)')
    parser.add_argument('-l', '--regularizes_lambda', default=0.01, type=float, help='L2 regulation parameter.(default=0.01)')
    parser.add_argument('-b', '--batch_size', default=64, type=int, help='Mini-Batch size.(default=64)')
    parser.add_argument('--epochs', default=10, type=int, help='Number of epochs.(default=10)')
    parser.add_argument('--fraction_validation', default=0.05, type=float, help='The fraction of validation.(default=0.05)')
    parser.add_argument('--results_dir', default='result/textCNN/', type=str, help='The results dir including log, model, vocabulary and some images.(default=result/textCNN/)')
    args = parser.parse_args()
    print('Parameters:', args, '\n')

    timestamp = time.strftime("%Y-%m-%d-%H-%M", time.localtime(time.time()))
    os.mkdir(os.path.join(args.results_dir, timestamp))
    os.mkdir(os.path.join(args.results_dir, timestamp, 'log/'))

    x_train, y_train, vocab_size = preprocess("data/sentiment_train.csv", args.padding_size)
                                
    train(x_train, y_train, vocab_size, args.padding_size, os.path.join(args.results_dir, timestamp))
