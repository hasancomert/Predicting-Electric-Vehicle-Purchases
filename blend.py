"""OOF tahminleri üzerinde hill climbing ile harman ağırlıklarını bulur.
Kullanım: python blend.py pevpsubmission3 lgbm_... xgb_... cat_...  ->  pevpsubmission3.csv
Virgülle ayrılmış etiketler (ör. tohumlar) önce ortalanır ve tek model sayılır."""
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

out, tags = sys.argv[1], sys.argv[2:]
y = (pd.read_csv("train.csv", usecols=["Will_Buy_EV"]).Will_Buy_EV == "Yes").to_numpy()
load = lambda kind, tag: np.mean([np.load(f"{kind}_{t}.npy") for t in tag.split(",")], axis=0)
oof = np.column_stack([load("oof", t) for t in tags])
pred = np.column_stack([load("pred", t) for t in tags])
single = [roc_auc_score(y, o) for o in oof.T]
for t, s in zip(tags, single):
    print(f"{s:.5f}  {t}")

# Her adımda harmana (tekrar seçilebilir) AUC'yi en çok artıran modeli ekle.
counts = np.eye(len(tags))[np.argmax(single)]
best = max(single)
while True:
    score, j = max((roc_auc_score(y, oof @ (counts + np.eye(len(tags))[j])), j)
                   for j in range(len(tags)))
    if score <= best + 1e-6:
        break
    best, counts[j] = score, counts[j] + 1

w = counts / counts.sum()
print("ağırlıklar:", {t: round(float(x), 3) for t, x in zip(tags, w) if x})
print(f"harman CV AUC: {best:.5f}")
sub = pd.read_csv("sample_submission.csv")
sub["Will_Buy_EV"] = pred @ w
sub.to_csv(f"{out}.csv", index=False)
print(f"{out}.csv yazıldı, {len(sub)} satır")
