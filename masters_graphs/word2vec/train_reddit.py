from train_strategy import node2vec_train_strategy

from dataset_reddit import RedditDataset

import logging


def main():
    walk_length = 2
    embedding_dim = 10
    logging.basicConfig(level=logging.INFO)
    logging.info("Starting loading dataset.")
    reddit_train = RedditDataset(
        walk_length=walk_length, dataset_dir="./data/reddit", dataset_mode="train"
    )
    logging.info("Loaded dataset.")
    node2vec_train_strategy(walk_length, embedding_dim, reddit_train)


if __name__ == "__main__":
    main()
