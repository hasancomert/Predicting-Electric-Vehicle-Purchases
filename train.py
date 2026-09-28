"""Özellik grupları + model, 5 katlı CV; oof_<etiket>.npy ve pred_<etiket>.npy yazar.
Kullanım: python train.py <lgbm|xgb|cat> <gruplar> [öğrenme oranı, varsayılan 0.1]
Gruplar (virgülle): base, freq, dig, dig2, recipe, omean, te1, te3, bins, bins2, te2, te2s, ted, ncat,
orig, mb (lgbm/xgb max_bin=1024) (README'de sonuçlar)
PVEP_GPU=1 ile xgb ve cat GPU'da eğitilir, etikete _gpu eklenir (kaggle_run.py bunu ayarlar)."""
import os
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
GPU = os.environ.get("PVEP_GPU") == "1" and model_name != "lgbm"  # pip LightGBM'i CUDA'sız

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
if "dig2" in groups:  # sayısal sütunların tek tek haneleri (10^-1..10^4), sabit ve kopya olanlar hariç
    for c in NUMS:
        for k in range(-1, 5):
            d = np.floor(np.round(allx[c] / 10.0 ** k, 6)) % 10
            if d.nunique() > 1 and not np.allclose(d.fillna(-1), allx[c].fillna(-1)):
                F[f"{c}_d{k}"] = d
if "recipe" in groups:  # orijinal verinin üretim formülleri, gürültüsüz kısım (C. Deotte'nin EDA'sı)
    anx = allx.Range_Anxiety_Level
    F["buy_score"] = (1.2 * allx.Annual_Income_USD / 1e5 + 0.6 * allx.Environmental_Concern_Level
                      + 2.0 * (allx.Subsidy_Available == "Yes") - 1.0 * (anx == "Medium")
                      - 3.0 * (anx == "High"))
    F["anx_score"] = (allx.Daily_Commute_km - 5 * allx.Charging_Stations_Near_Home
                      - 5 * allx.Charging_Stations_Near_Work
                      - 150.0 * (allx.Home_Charging_Possible == "Yes"))
if "omean" in groups:  # orijinal veride değer başına alım oranı (orada olmayan değere genel oran)
    for c in COLS:
        F[c + "_omean"] = allx[c].map(pd.Series(yo).groupby(orig[c]).mean()).fillna(yo.mean())

# hedef kodlama anahtarları: tekli (te1) ve ikili (te2) sütunlar
codes = {c: pd.factorize(allx[c])[0].astype(np.int64) + 1 for c in COLS}
keys = {}
if groups & {"te1", "te3", "te2", "te2s"}:
    keys.update(codes)
inc, com, bins = allx.Annual_Income_USD, allx.Daily_Commute_km, {}
if "bins" in groups:  # gelir /100, /1000 ve tam km: kaba ölçekte anahtarlar
    bins.update(inc_100=inc // 100, inc_1000=inc // 1000, com_int=com // 1)
if "bins2" in groups:  # ek ölçekler
    bins.update(inc_10=inc // 10, inc_500=inc // 500, inc_5000=inc // 5000, com_5=com // 5)
for k, v in bins.items():
    keys[k] = pd.factorize(v)[0].astype(np.int64) + 1
if "ted" in groups:  # hane izlerinin hedef kodlaması
    inc = allx.Annual_Income_USD.fillna(-1).astype(np.int64)
    keys.update(inc_mod100=inc % 100, inc_mod1000=inc % 1000,
                com_dec=((allx.Daily_Commute_km.fillna(-1) * 10).round() % 10).astype(int))
LOW = [c for c in COLS if c not in ("Annual_Income_USD", "Daily_Commute_km")]
if "te2" in groups or "te2s" in groups:
    for a, b in combinations(COLS if "te2" in groups else LOW, 2):
        keys[f"{a}|{b}"] = codes[a] * (codes[b].max() + 1) + codes[b]
K = np.column_stack(list(keys.values())) if keys else None
SMOOTH = ["auto", 10.0, 100.0] if "te3" in groups else ["auto"]  # hedef kodlama yumuşatmaları
MB = {"max_bin": 1024} if "mb" in groups else {}

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
    add = lambda d, e, names: pd.concat([d.reset_index(drop=True),
                                         pd.DataFrame(e, columns=names)], axis=1)
    for sm in (SMOOTH if K is not None else []):
        enc = TargetEncoder(target_type="binary", smooth=sm, cv=5, shuffle=True,
                            random_state=SEED)
        names = [f"te{'' if sm == 'auto' else int(sm)}_{k}" for k in keys]
        Xtr = add(Xtr, enc.fit_transform(K[idx_tr], ytr), names)
        Xva = add(Xva, enc.transform(K[va]), names)
        Xte = add(Xte, enc.transform(K[n:n + m]), names)
    return Xtr, ytr, Xva, Xte


def lgbm(Xtr, ytr, Xva, yva, Xte):
    mdl = lgb.LGBMClassifier(n_estimators=20000, learning_rate=LR, num_leaves=63,
                             min_child_samples=100, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.5, reg_lambda=1.0, random_state=SEED,
                             verbose=-1, **MB)
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(int(20 / LR), verbose=False)])
    return mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1], mdl.best_iteration_


def xgbm(Xtr, ytr, Xva, yva, Xte):
    mdl = xgb.XGBClassifier(n_estimators=20000, learning_rate=LR, max_depth=6,
                            min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
                            reg_lambda=1.0, tree_method="hist", device="cuda" if GPU else "cpu",
                            enable_categorical=True, max_cat_to_onehot=4, eval_metric="auc",
                            early_stopping_rounds=int(20 / LR), random_state=SEED, n_jobs=4, **MB)
    mdl.fit(Xtr, ytr, eval_set=[(Xva, yva)], verbose=False)
    return mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1], mdl.best_iteration


def cat(Xtr, ytr, Xva, yva, Xte):
    s = lambda d: d.astype({c: str for c in CATS})
    mdl = CatBoostClassifier(iterations=20000, learning_rate=LR, depth=6, eval_metric="AUC",
                             od_type="Iter", od_wait=int(20 / LR), cat_features=CATS,
                             task_type="GPU" if GPU else "CPU", border_count=254,  # GPU varsayılanı 128
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
tag = f"{model_name}_{'+'.join(sorted(groups))}_{LR}" + ("_gpu" if GPU else "")
print(f"{tag} CV AUC: {roc_auc_score(y, oof):.5f}  "
      f"[{time.time() - t0:.0f} sn, {Xtr.shape[1]} özellik]", flush=True)
np.save(f"oof_{tag}.npy", oof)
np.save(f"pred_{tag}.npy", pred)
