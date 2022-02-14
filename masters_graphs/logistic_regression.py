# dataset_to_load="Citeseer"
# dataset_to_load="Pubmed"
import pickle
from utils import MetricStore, load_dataset, print_dataset_summary
import torch
from sklearn.linear_model import LogisticRegression, SGDClassifier

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os

device = "cpu"

dataset_to_load, outdir = (
    os.environ["dataset_name"],
    os.environ["outdir_name"],
)
data, dataset = load_dataset(dataset_to_load, device)

masks = {
    "train": data.train_mask.to(device),
    "test": data.test_mask.to(device),
    "valid": data.val_mask.to(device),
}


pipe = make_pipeline(StandardScaler(), LogisticRegression())
# SGDClassifier(loss="log", n_jobs=-1)
# clf = LogisticRegression(solver='saga')
pipe.fit(data.x[data.train_mask].cpu(), data.y[data.train_mask].cpu())
predictions = pipe.predict_proba(data.x.cpu())
logistic_regression_store = MetricStore(masks=masks, num_classes=dataset.num_classes)
logistic_regression_store.calculate_all(predictions, data.y.cpu().detach().numpy())


print(round(logistic_regression_store.store["accuracy"]["test"][-1], 3))
print(logistic_regression_store.store["f1_score"]["test"][-1].round(3))
print(logistic_regression_store.store["gini"]["test"][-1].round(3))
with open(f"masters_graphs/{outdir}/logistic_regression_result.pickle", "wb") as f:
    pickle.dump(logistic_regression_store, f)
