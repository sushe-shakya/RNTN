"""
Utility functions
"""
from settings import BASE_DIR
import numpy as np
import pickle


def save_to_file(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / sum(e)


def get_glove_embeddings(dimension):
    word_embeddings = {}
    fd = open(f"{BASE_DIR}/models/glove.6B/glove.6B.{dimension}d.txt", 'r')
    lines = fd.readlines()
    for line in lines:
        tokens = line.strip().split(' ')
        word = tokens[0]
        embeddings = np.asarray([float(a) for a in tokens[1:]])
        word_embeddings[word] = embeddings
    return word_embeddings
