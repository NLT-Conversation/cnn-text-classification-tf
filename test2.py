from os import listdir
from os.path import isfile, join
import logging
import numpy as np
import re
from collections import Counter


def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def load_data(dir_name):
    onlyfiles = [f for f in listdir(dir_name) if isfile(join(dir_name, f))]
    y = []
    x_text = []
    labels = []
    for f in onlyfiles:
        f_data = list(open(dir_name + f, "r").readlines())
        f_data = [s.strip() for s in f_data]
        labels = [f.split(".txt")[0] for _ in f_data]
        x_text += f_data
        y = np.concatenate([y, labels], 0)
    labels_unique = sorted(list(set(y)))
    num_labels = len(labels_unique)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels_unique, one_hot))
    print num_labels, label_dict
    x_text = np.array([clean_str(sent) for sent in x_text])
    #x_text = pad_sentences(x_text)
    #vocabulary, vocabulary_inv = build_vocab(x_text)
    #x = np.array([[vocabulary[word] for word in sentence] for sentence in x_text])
    y = np.array([label_dict[i] for i in y])
    return x_text, y, label_dict
    # return x, y, vocabulary, vocabulary_inv, label_dict


def load_embeddings(vocabulary):
    word_embeddings = {}
    for word in vocabulary:
        word_embeddings[word] = np.random.uniform(-0.25, 0.25, 300)
    return word_embeddings


def pad_sentences(sentences, padding_word="<PAD/>", forced_sequence_length=None):
    """Pad setences during training or prediction"""
    if forced_sequence_length is None:  # Train
        sequence_length = max(len(x) for x in sentences)
    else:  # Prediction
        logging.critical('This is prediction, reading the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('The maximum length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        if num_padding < 0:  # Prediction: cut off the sentence if it is longer than the sequence length
            logging.info('This sentence has to be cut off because it is longer than trained sequence length')
            padded_sentence = sentence[0:sequence_length]
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences


def build_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


if __name__ == "__main__":
    train_file = './data/stance/'
    x, y, labels = load_data(train_file)
    print labels
