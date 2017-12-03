from os import listdir
from os.path import isfile, join
import numpy as np
import re


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


mypath = './data/stance/'
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
print onlyfiles
y = []
x_text = []
for f in onlyfiles:
    f_data = list(open(mypath + f, "r").readlines())
    f_data = [s.strip() for s in f_data]
    labels = [f for _ in f_data]

    x_text += f_data

    y = np.concatenate([y, labels], 0)

x_text = [clean_str(sent) for sent in x_text]

labels = sorted(list(set(y)))
one_hot = np.zeros((len(labels), len(labels)), int)
np.fill_diagonal(one_hot, 1)
label_dict = dict(zip(labels, one_hot))
print label_dict
y = [label_dict[x] for x in y]

print y
#print [x_text, y]
