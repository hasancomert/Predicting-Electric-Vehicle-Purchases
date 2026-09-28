"""Özellik grupları + model, 5 katlı CV; oof_<etiket>.npy ve pred_<etiket>.npy yazar.
Kullanım: python train.py <lgbm|xgb|cat> <gruplar> [öğrenme oranı, varsayılan 0.1]
Gruplar (virgülle): base, freq, dig, te1, te2, te2s, ted, ncat, orig (README'de sonuçlar)"""
import sys
import time
from itertools import combinations

import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import TargetEncoder

SEED, FOLDS, TARGET = 42, 5, "Will_Buy_EV"
CATS = ["Gender", "City_Type", "Current_Car_Type", "Home_Charging_Possible",
        "Subsidy_Available", "Range_Anxiety_Level"]
NUMS = ["Age", "Annual_Income_USD", "Daily_Commute_km", "Number_of_Cars_Owned",
        "Charging_Stations_Near_Home", "Charging_Stations_Near_Work",
        "Environmental_Concern_Level"]
COLS = NUMS + CATS

model_name, groups = sys.argv[1], set(sys.argv[2].split(","))
LR = float(sys.argv[3]) if len(sys.argv) > 3 else 0.1

train, test, orig = (pd.read_csv(f) for f in ("train.csv", "test.csv", "original.csv"))
y = (train[TARGET] == "Yes").to_numpy(int)
yo = (orig[TARGET] == "Yes").to_numpy(int)
n, m = len(train), len(test)
allx = pd.concat([train[COLS], test[COLS], orig[COLS]], ignore_index=True)

F = allx.copy()
for c in CATS:
    F[c] = pd.Categorical(F[c])
if "ncat" in groups:  # küçük tam sayı sütunlarının kategori kopyası
    for c in ["Age", "Number_of_Cars_Owned", "Charging_Stations_Near_Home",
              "Charging_Stations_Near_Work", "Environmental_Concern_Level"]:
        F[c + "_cat"] = pd.Categorical(allx[c])
if "freq" in groups:  # değerin veride kaç kez geçtiği (etiketsiz, sızıntı yok)
    for c in ["Age", "Annual_Income_USD", "Daily_Commute_km"]:
        F[c + "_freq"] = allx[c].map(allx[c].value_counts())
if "dig" in groups:  # gelir/mesafe hane izleri
    inc = allx.Annual_Income_USD
    F["inc_mod10"] = inc % 10
    F["inc_mod100"] = inc % 100
    F["inc_mod1000"] = inc % 1000
    F["com_dec"] = (allx.Daily_Commute_km * 10).round() % 10

# hedef kodlama anahtarları: tekli (te1) ve ikili (te2) sütunlar
codes = {c: pd.factorize(allx[c])[0].astype(np.int64) + 1 for c in COLS}
keys = {}
if groups & {"te1", "te2", "te2s"}:
    keys.update(codes)
if "ted" in groups:  # hane izlerinin hedef kodlaması
    inc = allx.Annual_Income_USD.fillna(-1).astype(np.int64)
    keys.update(inc_mod100=inc % 100, inc_mod1000=inc % 1000,
                com_dec=((allx.Daily_Commute_km.fillna(-1) * 10).round() % 10).astype(int))
LOW = [c for c in COLS if c not in ("Annual_Income_USD", "Daily_Commute_km")]
if "te2" in groups or "te2s" in groups:
    for a, b in combinations(COLS if "te2" in groups else LOW, 2):
        keys[f"{a}|{b}"] = codes[a] * (codes[b].max() + 1) + codes[b]
K = np.column_stack(list(keys.values())) if keys else None

X, X_test, X_orig = F.iloc[:n], F.iloc[n:n + m], F.iloc[n + m:]
folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y))
use_orig = "orig" in groups


def fold_data(tr, va):
    Xtr, ytr, idx_tr = X.iloc[tr], y[tr], tr
    if use_orig:
        Xtr = pd.concat([Xtr, X_orig])
        ytr = np.concatenate([ytr, yo])
        idx_tr = np.concatenate([tr, np.arange(n + m, n + m + len(orig))])
    Xva, Xte = X.iloc[va], X_test
    if K is not None:
        enc = TargetEncoder(target_type="binary", cv=5, shuffle=True, random_state=SEED)
        names = [f"te_{k}" for k in keys]
        add = lambda d, e: pd.concat([d.reset_index(drop=True),
                                      pd.DataFrame(e, columns=names)], axis=1)
        Xtr = add(Xtr, enc.fit_transform(K[idx_tr], ytr))
        Xva = add(Xva, enc.transform(K[va]))
        Xte = add(Xte, enc.transform(K[n:n + m]))
    return Xtr, ytr, Xva, Xte


def lgbm(Xtr, ytr, Xva, yva, Xte):
    mdl = lgb.LGBMClassifier(n_estimators=20000, learning_rate=LR, num_leaves=63,
                             min_child_samples=100, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, reg_lambda=1.0, random_state=SEED,
                             verbose=-1)
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(int(20 / LR), verbose=False)])
    return mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1], mdl.best_iteration_


def xgbm(Xtr, ytr, Xva, yva, Xte):
    mdl = xgb.XGBClassifier(n_estimators=20000, learning_rate=LR, max_depth=6,
                            min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
                            reg_lambda=1.0, tree_method="hist", enable_categorical=True,
                            max_cat_to_onehot=4, eval_metric="auc",
                            early_stopping_rounds=int(20 / LR), random_state=SEED, n_jobs=4)
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1], mdl.best_iteration


def cat(Xtr, ytr, Xva, yva, Xte):
    s = lambda d: d.astype({c: str for c in CATS})
    mdl = CatBoostClassifier(iterations=20000, learning_rate=LR, depth=6, eval_metric="AUC",
                             od_type="Iter", od_wait=int(20 / LR), cat_features=CATS,
                             random_seed=SEED, verbose=0, allow_writing_files=False)
    mdl.fit(s(Xtr), ytr, eval_set=(s(Xva), yva))
    return (mdl.predict_proba(s(Xva))[:, 1], mdl.predict_proba(s(Xte))[:, 1],
            mdl.get_best_iteration())


t0 = time.time()
oof, pred = np.zeros(n), np.zeros(m)
for i, (tr, va) in enumerate(folds):
    Xtr, ytr, Xva, Xte = fold_data(tr, va)
    oof[va], p, it = {"lgbm": lgbm, "xgb": xgbm, "cat": cat}[model_name](Xtr, ytr, Xva, y[va], Xte)
    pred += p / FOLDS
    print(f"  fold {i}: {roc_auc_score(y[va], oof[va]):.5f} ({it} it)", flush=True)
tag = f"{model_name}_{'+'.join(sorted(groups))}_{LR}"
print(f"{tag} CV AUC: {roc_auc_score(y, oof):.5f}  "
      f"[{time.time() - t0:.0f} sn, {Xtr.shape[1]} özellik]", flush=True)
np.save(f"oof_{tag}.npy", oof)
np.save(f"pred_{tag}.npy", pred)
