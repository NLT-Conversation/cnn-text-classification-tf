import numpy as np
import re
import itertools
from collections import Counter
from os import listdir
from os.path import isfile, join


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


# def load_data_and_labels(positive_data_file, negative_data_file):
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
    x_text = np.array([clean_str(sent) for sent in x_text]
    y=np.array([label_dict[i].toList() for i in y])
    return [x_text, y, label_dict]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data=np.array(data)
    data_size=len(data)
    num_batches_per_epoch=int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices=np.random.permutation(np.arange(data_size))
            shuffled_data=data[shuffle_indices]
        else:
            shuffled_data=data
        for batch_num in range(num_batches_per_epoch):
            start_index=batch_num * batch_size
            end_index=min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]
