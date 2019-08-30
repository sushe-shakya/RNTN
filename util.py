"""
Utility functions
"""
from bert_embedding import BertEmbedding
import numpy as np
import pickle

bert_embedding = BertEmbedding()


def save_to_file(model, filename):
    with open(filename, 'wb') as f:
        pickle.dump(model, f)


def load_from_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / sum(e)


def get_bert_embeddings(text):
    tokens, embeddings = bert_embedding([text])[0]
    token_embeddings_map = {}
    for i in range(len(tokens)):
        token_embeddings_map[tokens[i]] = embeddings[i]
    return token_embeddings_map

    # train_trees = load_trees(dataset='train')
