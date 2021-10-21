from __future__ import print_function
import json
import numpy as np

import networkx_nodelink_legacy as json_graph
from argparse import ArgumentParser
from sklearn.linear_model import SGDClassifier
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


def run_regression(train_embeds, train_labels, test_embeds, test_labels):
    np.random.seed(1)
    dummy = DummyClassifier()
    dummy.fit(train_embeds, train_labels)
    log = SGDClassifier(loss="log", n_jobs=55)
    log.fit(train_embeds, train_labels)
    print("Test scores")
    print(f1_score(test_labels, log.predict(test_embeds), average="micro"))
    print("Train scores")
    print(f1_score(train_labels, log.predict(train_embeds), average="micro"))
    print("Random baseline")
    print(f1_score(test_labels, dummy.predict(test_embeds), average="micro"))


def parse_args():
    parser = ArgumentParser("Run evaluation on Reddit data.")
    parser.add_argument("dataset_dir", help="Path to directory containing the dataset.")
    parser.add_argument(
        "embed_dir",
        help="Path to directory containing the learned node embeddings. Set to 'feat' for raw features.",
    )
    parser.add_argument("setting", help="Either val or test.")
    args = parser.parse_args()
    dataset_dir: str = args.dataset_dir
    data_dir: str = args.embed_dir
    setting: str = args.setting
    return dataset_dir, data_dir, setting


def load_data(dataset_dir, setting):

    print("Loading data...")
    with open(dataset_dir + "/reddit-G.json") as f:
        js = json.load(f)
        G = json_graph.node_link_graph(js)
    with open(dataset_dir + "/reddit-class_map.json") as f:
        labels = json.load(f)

    train_ids = [
        n
        for n in G.nodes()
        if not G.nodes(data=True)[n].get("val", False)
        and not G.nodes(data=True)[n].get("test", False)
    ]
    test_ids = [n for n in G.nodes() if G.nodes(data=True)[n].get(setting, False)]
    train_labels = [labels[i] for i in train_ids]
    test_labels = [labels[i] for i in test_ids]
    return train_ids, test_ids, train_labels, test_labels


def preprocess_features(dataset_dir, train_ids, test_ids):
    print("Using only features..")
    feats = np.load(dataset_dir + "/reddit-feats.npy")
    ## Logistic gets thrown off by big counts, so log transform num comments and score
    feats[:, 0] = np.log(feats[:, 0] + 1.0)
    feats[:, 1] = np.log(feats[:, 1] - min(np.min(feats[:, 1]), -1))
    feat_id_map = json.load(open(dataset_dir + "/reddit-id_map.json"))
    feat_id_map = {id: val for id, val in feat_id_map.items()}
    train_feats = feats[[feat_id_map[id] for id in train_ids]]
    test_feats = feats[[feat_id_map[id] for id in test_ids]]

    scaler = StandardScaler()
    scaler.fit(train_feats)
    train_feats = scaler.transform(train_feats)
    test_feats = scaler.transform(test_feats)
    return train_feats, test_feats


def main():
    dataset_dir, data_dir, setting = parse_args()
    train_ids, test_ids, train_labels, test_labels = load_data(dataset_dir, setting)

    if data_dir == "feat":
        train_feats, test_feats = preprocess_features(dataset_dir, train_ids, test_ids)
        print("Running regression..")
        run_regression(train_feats, train_labels, test_feats, test_labels)

    elif "n2v" in data_dir:
        print("Doing it N2V style.")
        base_embeds = np.load(data_dir + "/val.npy")
        base_id_map = {}
        with open(data_dir + "/val.txt") as fp:
            for i, line in enumerate(fp):
                base_id_map[line.strip()] = i
        tuned_embeds = np.load(data_dir + "/val-test.npy")
        tuned_id_map = {}
        with open(data_dir + "/val-test.txt") as fp:
            for i, line in enumerate(fp):
                tuned_id_map[line.strip()] = i
        train_embeds = base_embeds[[base_id_map[id] for id in train_ids]]
        test_embeds = tuned_embeds[[tuned_id_map[id] for id in test_ids]]

        print("Running regression..")
        run_regression(train_embeds, train_labels, test_embeds, test_labels)

        # loading feats
        feats = np.load(dataset_dir + "/reddit-feats.npy")
        feat_id_map = json.load(open(dataset_dir + "/reddit-id_map.json"))
        feat_id_map = {id: val for id, val in feat_id_map.iteritems()}
        train_feats = feats[[feat_id_map[id] for id in train_ids]]
        test_feats = feats[[feat_id_map[id] for id in test_ids]]
        train_embeds = np.hstack([train_feats, train_embeds])
        test_embeds = np.hstack([test_feats, test_embeds])
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler()
        scaler.fit(train_embeds)
        train_embeds = scaler.transform(train_embeds)
        test_embeds = scaler.transform(test_embeds)

        print("Running regression with feats..")
        run_regression(train_embeds, train_labels, test_embeds, test_labels)
    else:
        embeds = np.load(data_dir + "/val.npy")
        id_map = {}
        with open(data_dir + "/val.txt") as fp:
            for i, line in enumerate(fp):
                id_map[line.strip()] = i
        train_embeds = embeds[[id_map[id] for id in train_ids]]
        test_embeds = embeds[[id_map[id] for id in test_ids]]

        print("Running regression..")
        run_regression(train_embeds, train_labels, test_embeds, test_labels)


if __name__ == "__main__":
    main()
