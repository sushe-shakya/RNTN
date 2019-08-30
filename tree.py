#!/bin/env python3

from nltk.parse import CoreNLPParser
from nltk.tree import ParentedTree
from collections import defaultdict
import logging
import util
import os


UNK = 'unk'

WORD_MAP_FILENAME = 'models/word_map.pickle'
logger = logging.getLogger(__file__)


def parse(text):
    parser = CoreNLPParser("http://localhost:9000")
    result = parser.raw_parse(text.lower())
    trees = [tree for tree in result]
    for tree in trees:
        tree.chomsky_normal_form()
        tree.collapse_unary(collapseRoot=True, collapsePOS=True)
    trees = [ParentedTree.convert(tree) for tree in trees]
    return trees


def isleaf(tree):
    return isinstance(tree, ParentedTree) and tree.height() == 2


def traverse(tree, f=logger.info, args=None, leaves=False):
    if leaves:
        if isleaf(tree):
            f(tree, args)
            return
    else:
        f(tree, args)
        if isleaf(tree):
            return
    for child in tree:
        traverse(child, f, args)


def build_word_map():
    logger.info("Building word map...")
    with open("trees/train.txt", "r") as f:
        trees = [ParentedTree.fromstring(line.lower()) for line in f]

    logger.info("Counting words...")
    words = defaultdict(int)
    for tree in trees:
        for token in tree.leaves():
            words[token] += 1

    word_map = dict(zip(words.keys(), range(len(words))))
    word_map[UNK] = len(words)  # Add unknown as word
    util.save_to_file(word_map, WORD_MAP_FILENAME)
    return word_map


def load_word_map():
    if not os.path.isfile(WORD_MAP_FILENAME):
        return build_word_map()
    logger.info("Loading word map...")
    return util.load_from_file(WORD_MAP_FILENAME)


def load_trees(dataset='train'):
    filename = "trees/{}.txt".format(dataset)
    with open(filename, 'r') as f:
        logger.info("Reading '{}'...".format(filename))
        trees = [ParentedTree.fromstring(line.lower()) for line in f]
    return trees


def text_from_tree(tree, text=""):

    if isleaf(tree):
        token = str(tree).split(" ")[1].strip()[:-1]
        text = text + token + " "
        return text

    for child in tree:
        text = text_from_tree(child, text)

    return text


def setup_word_embeddings(trees):
    """Get embeddings for "hello" of
       sentence 10: word_embeddings["hello"][10]"""
    sentence_index = {}
    word_embeddings = {}
    for i, tree in enumerate(trees):
        text = text_from_tree(tree)
        sentence_index[text] = i
        sentence_token_embeddings = util.get_bert_embeddings(text)
        for token, embeddings in sentence_token_embeddings.items():
            if token not in word_embeddings:
                word_embeddings[token] = {}
                word_embeddings[token][i] = embeddings
            else:
                word_embeddings[token][i] = embeddings
        logger.info("Word embeddings setup completed for: {}".format(i))
    return word_embeddings, sentence_index


if __name__ == '__main__':
    word_map = load_word_map()
