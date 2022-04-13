import tensorflow as tf
from models.generator import Generator
from gan_config import BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH, pretrained_generator_file
import numpy as np

import os
import sys
# 项目路径,将项目路径保存
project_path = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_path)

result_dir = project_path + os.sep + 'result' + os.sep

physical_devices = tf.config.experimental.list_physical_devices("GPU")
if len(physical_devices) > 0:
    for dev in physical_devices:
        tf.config.experimental.set_memory_growth(dev, True)


def load_generator():
    generator = Generator(BATCH_SIZE, EMB_DIM, HIDDEN_DIM, SEQ_LENGTH)
    generator.load(pretrained_generator_file)
    return generator

def sequences_to_texts(sentences):
    word_dict = np.load(result_dir + 'w2dic.npy', allow_pickle=True).item()
    key_list = list(word_dict.keys())
    vocabulary = word_dict.values()
    value_list = list(vocabulary)
    results = []
    for each_sentence in sentences:
        x = ''
        for each_word in each_sentence:
            if each_word in vocabulary:
                x += key_list[value_list.index(each_word)]
            else:
                x += ' '
        results.append(x)
    return results

if __name__ == "__main__":
    generator = load_generator()
    generated_sentences = generator.generate_one_batch().numpy()
    sentences = sequences_to_texts(generated_sentences)
    print(*sentences, sep='\n')
