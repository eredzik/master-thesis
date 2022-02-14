# dataset_to_load="Citeseer"
# dataset_to_load="Pubmed"
import pickle
from utils import MetricStore, load_dataset
from sklearn.dummy import DummyClassifier
import os

device = "cpu"
# device='cuda:0' if torch.cuda.is_available() else 'cpu'

dataset_to_load, outdir = os.environ["dataset_name"], os.environ["outdir_name"]
# dataset_to_load, outdir = "Pubmed", "pubmed"
# dataset_to_load = "Reddit"
data, dataset = load_dataset(dataset_to_load, device)

# print_dataset_summary(dataset, data, dataset_to_load)
masks = {
    "train": data.train_mask.to(device),
    "test": data.test_mask.to(device),
    "valid": data.val_mask.to(device),
}

# free_gpu_cache()

clf = DummyClassifier()
clf.fit(data.x[data.train_mask].cpu(), data.y[data.train_mask].cpu())
predictions = clf.predict_proba(data.x.cpu())
store = MetricStore(masks=masks, num_classes=dataset.num_classes)
store.calculate_all(predictions, data.y.cpu().detach().numpy())

print(store.store["gini"]["test"][-1].round(3))
print(round(store.store["accuracy"]["test"][-1], 3))
print(store.store["f1_score"]["test"][-1].round(3))
with open(f"masters_graphs/{outdir}/dummy_regression_result.pickle", "wb") as f:
    pickle.dump(store, f)
