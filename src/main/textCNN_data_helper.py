import re
import pandas as pd
from tensorflow.keras import preprocessing
import numpy as np
import json

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

def preprocess(data_file, vocab_file, padding_size, test=False):
    """
    Text to sequence, compute vocabulary size, padding sequence.
    Return sequence and label.
    """
    print("Loading data from {} ...".format(data_file))
    df = pd.read_csv(data_file, header=None, names=["labels", "text"])
    y, x_text = df["labels"].tolist(), df["text"].tolist()

    if not test:
        # Texts to sequences TODO 使用真实的word2vec字典
        text_preprocessor = preprocessing.text.Tokenizer(oov_token="<UNK>")
        text_preprocessor.fit_on_texts(x_text)
        x = text_preprocessor.texts_to_sequences(x_text)
        word_dict = text_preprocessor.word_index
        json.dump(word_dict, open(vocab_file, 'w'), ensure_ascii=False)
        vocab_size = len(word_dict)
        # max_doc_length = max([len(each_text) for each_text in x])
        x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size,
                                                 padding='post', truncating='post')
        print("Vocabulary size: {:d}".format(vocab_size))
        print("Shape of train data: {}".format(np.shape(x)))
        return x, y, vocab_size + 1
    else:
        word_dict = json.load(open(vocab_file, 'r'))
        vocabulary = word_dict.keys()
        x = [[word_dict[each_word] if each_word in vocabulary else 1 for each_word in each_sentence.split()] for each_sentence in x_text]
        x = preprocessing.sequence.pad_sequences(x, maxlen=padding_size,
                                                 padding='post', truncating='post')
        print("Shape of test data: {}\n".format(np.shape(x)))
        return x, y