import re
import pandas as pd
from tensorflow.keras import preprocessing
import numpy as np
import os

result_dir = 'result' + os.sep

def text_preprocess(text):
    """
    Clean and segment the text.
    Return a new text.
    """
    text = re.sub(r"[\d+\s+\.!\/_,?=\$%\^\)*\(\+\"\'\+——！:；，。？、~@#%……&*（）·¥\-\|\\《》〈〉～]",
                  "", text)
    text = re.sub("[<>]", "", text)
    text = re.sub("[a-zA-Z0-9]", "", text)
    text = re.sub(r"\s", "", text)
    if not text:
        return ''
    return ' '.join(string for string in text)

def preprocess(data_file, padding_size, test=False):
    """
    Text to sequence, compute vocabulary size, padding sequence.
    Return sequence and label.
    """
    print("Loading data from {} ...".format(data_file))
    df = pd.read_csv(data_file, header=None, names=["labels", "text"])
    y, x_text = df["labels"].tolist(), df["text"].tolist()
    word_dict = np.load(result_dir + 'w2dic.npy', allow_pickle=True).item()
    vocabulary = word_dict.keys()
    x = [[word_dict[each_word] if each_word in vocabulary else 0 for each_word in each_sentence.split()] for each_sentence in x_text]
    x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size,
                                             padding='post', truncating='post')

    if not test:
        # Texts to sequences
        vocab_size = len(word_dict)
        print("Vocabulary size: {:d}".format(vocab_size))
        print("Shape of train data: {}".format(np.shape(x)))
        return x, y, vocab_size + 1
    else:
        print("Shape of test data: {}\n".format(np.shape(x)))
        return x, y