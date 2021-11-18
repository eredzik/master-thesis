from train_strategy import node2vec_train_strategy

from dataset_karate import KarateClubDataset


def main():
    walk_length = 2
    embedding_dim = 10
    karate = KarateClubDataset(walk_length)
    node2vec_train_strategy(walk_length, embedding_dim, karate)


if __name__ == "__main__":
    main()
