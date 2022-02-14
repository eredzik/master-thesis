import pickle
from utils import MetricStore, load_dataset
from sklearn.neural_network import MLPClassifier
import os

device = "cpu"
# device='cuda:0' if torch.cuda.is_available() else 'cpu'


dataset_to_load, outdir, n_layers, n_neurons, activation = (
    os.environ["dataset_name"],
    os.environ["outdir_name"],
    int(os.environ["ann_n_layers"]),
    int(os.environ["ann_n_neurons"]),
    os.environ["ann_activation"],
)
data, dataset = load_dataset(dataset_to_load, device)
masks = {
    "train": data.train_mask.to(device),
    "test": data.test_mask.to(device),
    "valid": data.val_mask.to(device),
}

clf = MLPClassifier(
    hidden_layer_sizes=tuple([n_neurons] * n_layers), activation=activation
)

clf.fit(data.x[data.train_mask].cpu(), data.y[data.train_mask].cpu())

model_name = "Sieć neuronowa"
predictions = clf.predict_proba(data.x.cpu())
store = MetricStore(masks=masks, num_classes=dataset.num_classes)
store.calculate_all(predictions, data.y.cpu().detach().numpy())

print(round(store.store["accuracy"]["test"][-1], 3))
print(store.store["f1_score"]["test"][-1].round(3))
print(store.store["gini"]["test"][-1].round(3))

with open(f"masters_graphs/{outdir}/ann_regression.pickle", "wb") as f:
    pickle.dump(store, f)
