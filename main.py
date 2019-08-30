#!/bin/env python3

import argparse
import logging
import rntn
import tree as tr
import os
import pickle

LOGGING_LEVELS = {'critical': logging.CRITICAL,
                  'error': logging.ERROR,
                  'warning': logging.WARNING,
                  'info': logging.INFO,
                  'debug': logging.DEBUG}

log_level = os.getenv("LOG_LEVEL", "info")
logging_level = LOGGING_LEVELS[log_level]
logging.basicConfig(level=logging_level,
                    format='%(asctime)s %(levelname)s: %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S')

DATA_DIR = "trees"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
logger = logging.getLogger(__file__)


def main():

    # Parse arguments
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--dim", type=int, default=25,
                        help="Vector dimension")
    parser.add_argument("-k", "--output-dim", type=int, default=5,
                        help="Number of output classes")
    parser.add_argument("-e", "--epochs", type=int, default=50,
                        help="Maximum number of epochs")
    parser.add_argument("-f", "--dataset", type=str, default="train",
                        choices=['train', 'dev', 'test'], help="Dataset")
    parser.add_argument("-l", "--learning-rate", type=float, default=1e-2,
                        help="Learning rate")
    parser.add_argument("-b", "--batch-size", type=int, default=30,
                        help="Batch size")
    parser.add_argument("-r", "--reg", type=float, default=1e-6,
                        help="Regularization parameter")
    parser.add_argument("-t", "--test", action="store_true",
                        help="Test a model")
    parser.add_argument("-m", "--model", type=str,
                        default='models/RNTN.pickle',
                        help="Model file")
    parser.add_argument("-o", "--optimizer", type=str, default='adagrad',
                        help="Optimizer", choices=['sgd', 'adagrad'])
    args = parser.parse_args()

    # Test
    if args.test:
        logger.info("Testing...")
        model = rntn.RNTN.load(args.model)
        test_trees = tr.load_trees(args.dataset)
        cost, result = model.test(test_trees)
        accuracy = 100.0 * result.trace() / result.sum()
        logger.info("Cost = {:.2f}, Correct = {:.0f} / {:.0f}, Accuracy = {:.2f} %".format(
            cost, result.trace(), result.sum(), accuracy))
    else:
        # load the trees
        train_trees = tr.load_trees(args.dataset)

        WORD_EMEDDINGS_PATH = f"{BASE_DIR}/trees/{args.dataset}_word_embeddings.pkl"
        SENTENCE_INDEX_PATH = f"{BASE_DIR}/trees/{args.dataset}_sentence_index.pkl"

        if os.path.exists(WORD_EMEDDINGS_PATH) and \
                os.path.exists(SENTENCE_INDEX_PATH):
            logger.info("Loading word embeddings and sentence_index")
            fd = open(WORD_EMEDDINGS_PATH, 'rb')
            word_embeddings = pickle.load(fd)
            fd.close()

            fd = open(SENTENCE_INDEX_PATH, 'rb')
            sentence_index = pickle.load(fd)
            fd.close()

            logger.info("Loaded word embeddings and sentence_index")

        else:
            logger.info("Creating word embeddings and sentence_index")
            word_embeddings, sentence_index = \
                tr.setup_word_embeddings(train_trees)

            fd = open(WORD_EMEDDINGS_PATH, 'wb')
            pickle.dump(word_embeddings, fd)
            fd.close()

            fd = open(SENTENCE_INDEX_PATH, 'wb')
            pickle.dump(sentence_index, fd)
            fd.close()
            logger.info("Created word embeddings and sentence_index")

        # Initialize the model

        model = rntn.RNTN(
            dim=args.dim, output_dim=args.output_dim,
            batch_size=args.batch_size, reg=args.reg,
            learning_rate=args.learning_rate, max_epochs=args.epochs,
            optimizer=args.optimizer,
            word_embeddings=word_embeddings,
            sentence_index=sentence_index)

        # Train
        model.fit(train_trees, export_filename=args.model)


if __name__ == '__main__':
    main()
