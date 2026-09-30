"""Özellik grupları + model, 5 katlı CV; oof_<etiket>.npy ve pred_<etiket>.npy yazar.
Kullanım: python train.py <lgbm|xgb|cat|nn> <gruplar> [öğrenme oranı, varsayılan 0.1] [ayarlar]
nn: PyTorch MLP (sayısallar normal kantile, kategorikler gömme); lr ~0.002, ayarlar d, layers, drop,
emb, epochs, bs, wd, lowcat (en çok bu kadar farklı değerli sayısallara da gömme).
Ayarlar (virgülle, ör. max_depth=7,subsample=0.9) modelin varsayılanlarını ezer ve etikete eklenir;
seed=N model ve hedef kodlama tohumu (katlar sabit), folds=N kat sayısı (varsayılan 5), name=ad etikette
ayarların yerine geçer, trials=N Optuna ile N deneme arar (dosya yazmaz, en iyi ayarları basar; şimdilik xgb),
full=f CV'den sonra tüm train ile yeniden eğitir (katların ortalama ağaç sayısı x f), test tahminini
pred_<etiket>_full.npy'ye yazar; hold=h tam veri benzetimi: train'in 10 katlı bölmesinin h. katı etiketli
sahte test olur, kalan %90 ile CV (folds=9 önerilir) ve full=f; sahte testte kat ortalaması, tam veri
(x0.9-1.3 ağaç) ve sıra karışımlarının AUC'si basılır (etikete _h<h> eklenir); solo=1 lgbm/xgb'de her
ağacı tek sütunla sınırlar (interaction_constraints), pl=w her katta ikinci bir model eğitir: test
satırları birinci aşamanın olasılıklarıyla yumuşak etiketli (ağırlık w) eğitime eklenir.
Gruplar (virgülle): base, freq, freq2, dig, dig2, recipe, omean, flag, te1, te3, bins, bins2, tedig, te2,
te2s, ted, ncat, orig, mb (lgbm/xgb max_bin=1024), tok, tokx, mix, chain (gpt2_income_tokens.csv gerekir) (README'de
sonuçlar)
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


def num(v):  # "7" -> 7, "0.9" -> 0.9, diğerleri metin
    for t in (int, float):
        try:
            return t(v)
        except ValueError:
            pass
    return v


OVR = dict(kv.split("=") for kv in sys.argv[4].split(",")) if len(sys.argv) > 4 else {}
OVR = {k: num(v) for k, v in OVR.items()}
TRIALS, NAME, MSEED = OVR.pop("trials", 0), OVR.pop("name", None), OVR.pop("seed", SEED)
FOLDS = OVR.pop("folds", FOLDS)  # 10 katta tohum 42 megayak OOF kütüphanesiyle aynı bölme
PL = OVR.pop("pl", 0)  # pl=w: ikinci aşama, test satırlarını birinci aşamanın olasılıklarıyla (ağırlık w) görür
FULL = OVR.pop("full", 0)  # full=f: CV sonrası tüm train ile, katların ortalama ağaç sayısı x f ile yeniden eğit
MARGIN = OVR.pop("margin", 0)  # margin=1: GLR logit'inden başlayan artık model (glr_margins.npz, 10 kat tohum 42)
HOLD = OVR.pop("hold", -1)  # hold=h: 10 katlı bölmenin h. katı etiketli sahte test (tam veri benzetimi)
FULLNIT = OVR.pop("fullnit", 0)  # fullnit=N: CV yok, yalnız tüm train ile N ağaçlık model (ağaç sayısı önceden biliniyorsa)
GPU = os.environ.get("PVEP_GPU") == "1" and model_name != "lgbm"  # pip LightGBM'i CUDA'sız

train, test, orig = (pd.read_csv(f) for f in ("train.csv", "test.csv", "original.csv"))
extra = test.iloc[:0]
if HOLD >= 0:  # gerçek test yalnız frekanslar aynı kalsın diye allx'in sonunda durur
    yall = (train[TARGET] == "Yes").to_numpy(int)
    keep, hold = list(StratifiedKFold(10, shuffle=True, random_state=SEED).split(train, yall))[HOLD]
    extra, test, train = test, train.iloc[hold].reset_index(drop=True), train.iloc[keep].reset_index(drop=True)
    y_hold = yall[hold]
y = (train[TARGET] == "Yes").to_numpy(int)
yo = (orig[TARGET] == "Yes").to_numpy(int)
n, m = len(train), len(test)
allx = pd.concat([train[COLS], test[COLS], orig[COLS], extra[COLS]], ignore_index=True)

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
DIG = {}  # sayısal sütunların tek tek haneleri (10^-1..10^4), sabit ve kopya olanlar hariç
if groups & {"dig2", "tedig", "freq2"}:
    for c in NUMS:
        for k in range(-1, 5):
            d = np.floor(np.round(allx[c] / 10.0 ** k, 6)) % 10
            if d.nunique() > 1 and not np.allclose(d.fillna(-1), allx[c].fillna(-1)):
                DIG[f"{c}_d{k}"] = d
if "dig2" in groups:
    for k, d in DIG.items():
        F[k] = d
if "freq2" in groups:  # tüm sütunların ve hanelerin normalize frekansı
    for k, v in {**{c: allx[c] for c in COLS}, **DIG}.items():
        F[k + "_fq"] = v.map(v.value_counts(normalize=True))
if "flag" in groups:  # gelir eşik bölgeleri ve çevre ilgisi 1
    inc = allx.Annual_Income_USD
    F["is_30k"] = (inc == 30000).astype(float)
    F["is_cliff"] = (inc >= 170537).astype(float)
    F["is_dead"] = ((inc >= 38000) & (inc <= 42000)).astype(float)
    F["is_env1"] = (allx.Environmental_Concern_Level == 1).astype(float)
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
if "tedig" in groups:  # hanelerin hedef kodlaması
    bins.update(DIG)
TOK = {}  # gelirin GPT-2 BPE parçaları (veri üreticisi GPT-2 tabanlı olabilir; P. B. Elefante'nin fikri)
if groups & {"tok", "tokx", "chain"} or "mix" in groups and os.path.exists("gpt2_income_tokens.csv"):
    tk = pd.read_csv("gpt2_income_tokens.csv")  # scratchpad/k_gpt2tok ile Kaggle'da üretildi
    tk = tk.set_index(tk.value.astype(np.int64))
    iv = inc.fillna(-1).astype(np.int64)
    TOK = {k: pd.factorize(iv.map(v))[0].astype(np.int64) + 1
           for k, v in {"L1": tk.t1, "L2": tk.t1 * 60000 + tk.t2, "LAST": tk.ntok * 60000 + tk["last"]}.items()}
if "tok" in groups:  # ilk parça, ilk iki parça, son parça: hedef kodlama anahtarı ve (etiketsiz) sayım
    bins.update({f"tok_{k}": v for k, v in TOK.items()})
    for k, v in TOK.items():
        F[f"tok_{k}_cnt"] = np.log1p(np.bincount(v)[v])
    F["tok_n"] = iv.map(tk.ntok).fillna(0)
if "tokx" in groups:  # parça x bağlam hücreleri
    for t in ("L1", "L2"):
        for c in ["City_Type", "Home_Charging_Possible", "Subsidy_Available", "Environmental_Concern_Level",
                  "Range_Anxiety_Level"]:
            bins[f"tok_{t}|{c}"] = TOK[t] * (codes[c].max() + 1) + codes[c]
if "mix" in groups:  # aynı anahtarı paylaşan train+test satırlarının diğer sütun ortalamaları (kendisi hariç)
    gen = np.ones(len(allx))
    gen[n + m:n + m + len(orig)] = 0  # orijinal veri sayılmaz
    MV = {"env": allx.Environmental_Concern_Level, "sub": allx.Subsidy_Available == "Yes",
          "anx": allx.Range_Anxiety_Level.map({"Low": 0, "Medium": 1, "High": 2}),
          "home": allx.Home_Charging_Possible == "Yes", "urban": allx.City_Type == "Urban",
          "rural": allx.City_Type == "Rural", "suv": allx.Current_Car_Type == "SUV",
          "truck": allx.Current_Car_Type == "Truck", "sedan": allx.Current_Car_Type == "Sedan",
          "male": allx.Gender == "Male", "cars": allx.Number_of_Cars_Owned, "age": allx.Age,
          "sth": allx.Charging_Stations_Near_Home, "stw": allx.Charging_Stations_Near_Work, "cmt": com,
          "inc": inc / 1e4}
    mk = {"IV": codes["Annual_Income_USD"], "CV": codes["Daily_Commute_km"],
          "IVCV": pd.factorize(codes["Annual_Income_USD"] * (codes["Daily_Commute_km"].max() + 1)
                               + codes["Daily_Commute_km"])[0]}
    mk.update({k: TOK[k] for k in ("L1", "L2") if k in TOK})
    for kn, key in mk.items():
        cnt = np.bincount(key, weights=gen)[key]
        F[f"mix_{kn}_logn"] = np.log(np.maximum(cnt, 1))
        for vn, v in MV.items():
            if vn == "inc" and kn in ("IV", "L1", "L2", "IVCV") or vn == "cmt" and kn in ("CV", "IVCV"):
                continue
            v = v.to_numpy(float)
            gm = v[gen == 1].mean()
            F[f"mix_{kn}_{vn}"] = (np.bincount(key, weights=v * gen)[key] - v * gen + 5 * gm) / (cnt - gen + 5) - gm
for k, v in bins.items():
    keys[k] = pd.factorize(v)[0].astype(np.int64) + 1
CHAINS = []  # chain: hiyerarşik hedef oranı (gelir: ilk parça -> ilk iki parça -> tam değer; mesafe: tam km -> değer)
if "chain" in groups:
    CI = pd.factorize(com // 1)[0].astype(np.int64) + 1
    CHAINS = ([([TOK["L1"], TOK["L2"], codes["Annual_Income_USD"]], S) for S in (5, 20, 80)]
              + [([CI, codes["Daily_Commute_km"]], S) for S in (5, 20)])
    CHAIN_NAMES = [f"chain{j}_S{S}_{i}" for j, (ks, S) in enumerate(CHAINS) for i in range(len(ks))]
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

X, X_test, X_orig = F.iloc[:n], F.iloc[n:n + m], F.iloc[n + m:n + m + len(orig)]
folds = list(StratifiedKFold(FOLDS, shuffle=True, random_state=SEED).split(X, y))
use_orig = "orig" in groups


def chain_feats(fit, app):  # fit: etiketi kullanılan train satırları, app: kodlanan satırlar (allx dizinleri)
    out, prior = [], y[fit].mean()
    for ks, S in CHAINS:
        post = np.full(len(app), prior)
        for key in ks:
            sz = int(key.max()) + 1
            s_, c_ = np.bincount(key[fit], weights=y[fit], minlength=sz), np.bincount(key[fit], minlength=sz)
            post = (s_[key[app]] + S * post) / (c_[key[app]] + S)
            p = np.clip(post, 1e-6, 1 - 1e-6)
            out.append(np.log(p / (1 - p)))
    return np.column_stack(out)


def fold_data(tr, va):
    Xtr, ytr, idx_tr = X.iloc[tr], y[tr], tr
    if use_orig:
        Xtr = pd.concat([Xtr, X_orig])
        ytr = np.concatenate([ytr, yo])
        idx_tr = np.concatenate([tr, np.arange(n + m, n + m + len(orig))])
    Xva, Xte = (None if va is None else X.iloc[va]), X_test  # va=None: tam veri eğitimi
    add = lambda d, e, names: pd.concat([d.reset_index(drop=True),
                                         pd.DataFrame(e, columns=names)], axis=1)
    for sm in (SMOOTH if K is not None else []):
        enc = TargetEncoder(target_type="binary", smooth=sm, cv=5, shuffle=True,
                            random_state=MSEED)
        names = [f"te{'' if sm == 'auto' else int(sm)}_{k}" for k in keys]
        Xtr = add(Xtr, enc.fit_transform(K[idx_tr], ytr), names)
        if va is not None:
            Xva = add(Xva, enc.transform(K[va]), names)
        Xte = add(Xte, enc.transform(K[n:n + m]), names)
    if CHAINS:  # train satırları iç 5 katla çapraz, doğrulama ve test tüm eğitim satırlarıyla
        assert not use_orig
        ctr = np.zeros((len(tr), len(CHAIN_NAMES)))
        for a, b in StratifiedKFold(5, shuffle=True, random_state=MSEED).split(tr, y[tr]):
            ctr[b] = chain_feats(tr[a], tr[b])
        Xtr = add(Xtr, ctr, CHAIN_NAMES)
        if va is not None:
            Xva = add(Xva, chain_feats(tr, va), CHAIN_NAMES)
        Xte = add(Xte, chain_feats(tr, np.arange(n, n + m)), CHAIN_NAMES)
    return Xtr, ytr, Xva, Xte


def solo(p, cols, names=False):  # solo=1: her ağaç tek sütun (toplamsal model, sütun birleşimi yok)
    groups = [[c] for c in cols] if names else [[i] for i in range(len(cols))]  # xgb sütun adı ister
    return {"interaction_constraints": groups} if p.pop("solo", 0) else {}


top = lambda nit: max(nit) if isinstance(nit, list) else nit  # nit listesi: en büyüğüyle bir kez eğit
many = lambda nit, f: np.column_stack([f(k) for k in nit]) if isinstance(nit, list) else f(nit)


def lgbm(Xtr, ytr, Xva, yva, Xte, w=None, nit=None, mg=None, **p):  # nit: sabit ağaç sayısı; mg: başlangıç logit'leri
    p = p | solo(p, Xtr.columns)
    mdl = lgb.LGBMClassifier(**dict(n_estimators=20000, learning_rate=LR, num_leaves=63,
                                    min_child_samples=100, subsample=0.8, subsample_freq=1,
                                    colsample_bytree=0.5, reg_lambda=1.0, random_state=MSEED,
                                    verbose=-1, **MB) | p | ({"n_estimators": top(nit)} if nit else {}))
    if nit:
        mdl.fit(Xtr, ytr, sample_weight=w)
        return None, many(nit, lambda k: mdl.predict_proba(Xte, num_iteration=k)[:, 1]), nit
    if mg is not None:  # artık model: GLR logit'i init_score, tahmin = sigmoid(ham skor + logit)
        mdl.fit(Xtr, ytr, sample_weight=w, init_score=mg[0], eval_set=[(Xva, yva)], eval_init_score=[mg[1]],
                eval_metric="auc", callbacks=[lgb.early_stopping(int(20 / LR), verbose=False)])
        sg = lambda X, m: 1 / (1 + np.exp(-(mdl.predict(X, raw_score=True) + m)))
        return sg(Xva, mg[1]), sg(Xte, mg[2]), mdl.best_iteration_
    mdl.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], eval_metric="auc",
            callbacks=[lgb.early_stopping(int(20 / LR), verbose=False)])
    return mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1], mdl.best_iteration_


def xgbm(Xtr, ytr, Xva, yva, Xte, w=None, nit=None, mg=None, **p):
    p = p | solo(p, Xtr.columns, names=True) | ({"n_estimators": top(nit), "early_stopping_rounds": None} if nit else {})
    mdl = xgb.XGBClassifier(**dict(n_estimators=20000, learning_rate=LR, max_depth=6,
                                   min_child_weight=5, subsample=0.8, colsample_bytree=0.5,
                                   reg_lambda=1.0, tree_method="hist",
                                   device="cuda" if GPU else "cpu", enable_categorical=True,
                                   max_cat_to_onehot=4, eval_metric="auc",
                                   early_stopping_rounds=int(20 / LR), random_state=MSEED,
                                   n_jobs=4, **MB) | p)
    if nit:
        mdl.fit(Xtr, ytr, sample_weight=w, verbose=False)
        return None, many(nit, lambda k: mdl.predict_proba(Xte, iteration_range=(0, k))[:, 1]), nit
    if mg is not None:  # artık model: GLR logit'i base_margin
        mdl.fit(Xtr, ytr, sample_weight=w, base_margin=mg[0], eval_set=[(Xva, yva)], base_margin_eval_set=[mg[1]],
                verbose=False)
        return (mdl.predict_proba(Xva, base_margin=mg[1])[:, 1], mdl.predict_proba(Xte, base_margin=mg[2])[:, 1],
                mdl.best_iteration + 1)
    mdl.fit(Xtr, ytr, sample_weight=w, eval_set=[(Xva, yva)], verbose=False)
    return mdl.predict_proba(Xva)[:, 1], mdl.predict_proba(Xte)[:, 1], mdl.best_iteration + 1


def cat(Xtr, ytr, Xva, yva, Xte, w=None, nit=None, **p):
    s = lambda d: d.astype({c: str for c in CATS})
    q = dict(iterations=20000, learning_rate=LR, depth=6, eval_metric="AUC",
             od_type="Iter", od_wait=int(20 / LR), cat_features=CATS,
             task_type="GPU" if GPU else "CPU",
             border_count=254,  # GPU varsayılanı 128
             random_seed=MSEED, verbose=0, allow_writing_files=False) | p
    if nit:
        q = {k: v for k, v in q.items() if k not in ("od_type", "od_wait")} | {"iterations": top(nit)}
        mdl = CatBoostClassifier(**q)
        mdl.fit(s(Xtr), ytr, sample_weight=w)
        return None, many(nit, lambda k: mdl.predict_proba(s(Xte), ntree_end=k)[:, 1]), nit
    mdl = CatBoostClassifier(**q)
    mdl.fit(s(Xtr), ytr, sample_weight=w, eval_set=(s(Xva), yva))
    return (mdl.predict_proba(s(Xva))[:, 1], mdl.predict_proba(s(Xte))[:, 1],
            mdl.get_best_iteration() + 1)


def nn(Xtr, ytr, Xva, yva, Xte, **p):
    import torch
    from sklearn.preprocessing import QuantileTransformer
    q = dict(d=512, layers=3, drop=0.2, emb=8, epochs=12, bs=2048, wd=1e-4, lowcat=0) | p
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(MSEED)
    cats = [c for c in Xtr.columns if isinstance(Xtr[c].dtype, pd.CategoricalDtype)]
    nums = [c for c in Xtr.columns if c not in cats]
    qt = QuantileTransformer(n_quantiles=1000, output_distribution="normal", subsample=200_000,
                             random_state=MSEED).fit(Xtr[nums])
    low = {c: np.sort(Xtr[c].unique()) for c in nums if Xtr[c].nunique() <= q["lowcat"]}
    cards = [len(Xtr[c].cat.categories) for c in cats] + [len(v) for v in low.values()]
    def lcode(x, v):  # değerin v içindeki sırası + 1, görülmemiş değer 0
        i = np.clip(np.searchsorted(v, x), 0, len(v) - 1)
        return np.where(v[i] == x, i + 1, 0)
    codes = lambda d: np.stack([d[c].cat.codes.to_numpy() + 1 for c in cats]
                               + [lcode(d[c].to_numpy(), v) for c, v in low.items()], 1)
    tens = lambda d: (torch.tensor(qt.transform(d[nums]), dtype=torch.float32, device=dev),
                      torch.tensor(codes(d), dtype=torch.long, device=dev))
    (tn, tc), (vn, vc), (en, ec) = tens(Xtr), tens(Xva), tens(Xte)
    yt = torch.tensor(ytr, dtype=torch.float32, device=dev)

    class Net(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embs = torch.nn.ModuleList(torch.nn.Embedding(k + 1, q["emb"]) for k in cards)
            w, layers = len(nums) + q["emb"] * len(cards), []
            for _ in range(q["layers"]):
                layers += [torch.nn.Linear(w, q["d"]), torch.nn.BatchNorm1d(q["d"]),
                           torch.nn.SiLU(), torch.nn.Dropout(q["drop"])]
                w = q["d"]
            self.mlp = torch.nn.Sequential(*layers, torch.nn.Linear(w, 1))

        def forward(self, xn, xc):
            e = [m(xc[:, i]) for i, m in enumerate(self.embs)]
            return self.mlp(torch.cat([xn, *e], 1)).squeeze(1)

    net = Net().to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=LR, weight_decay=q["wd"])
    steps = -(-len(yt) // q["bs"])
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=q["epochs"] * steps)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    def predict(xn, xc):
        net.eval()
        with torch.no_grad():
            out = [torch.sigmoid(net(xn[i:i + 65536], xc[i:i + 65536]))
                   for i in range(0, len(xn), 65536)]
        return torch.cat(out).cpu().numpy()

    best, best_ep, state = -1, 0, None
    for ep in range(q["epochs"]):
        net.train()
        perm = torch.randperm(len(yt), device=dev)
        for i in range(steps):
            b = perm[i * q["bs"]:(i + 1) * q["bs"]]
            opt.zero_grad()
            loss_fn(net(tn[b], tc[b]), yt[b]).backward()
            opt.step()
            sched.step()
        auc = roc_auc_score(yva, predict(vn, vc))
        if auc > best:  # doğrulama AUC'si en iyi epoch'un ağırlıkları
            best, best_ep = auc, ep + 1
            state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(state)
    return predict(vn, vc), predict(en, ec), best_ep


MODELS = {"lgbm": lgbm, "xgb": xgbm, "cat": cat, "nn": nn}
SPACE = {  # Optuna arama uzayları (öğrenme oranı komut satırındaki)
    "xgb": lambda t: dict(
        max_depth=t.suggest_int("max_depth", 4, 10),
        min_child_weight=t.suggest_float("min_child_weight", 1, 100, log=True),
        subsample=t.suggest_float("subsample", 0.5, 1.0),
        colsample_bytree=t.suggest_float("colsample_bytree", 0.2, 1.0),
        reg_lambda=t.suggest_float("reg_lambda", 0.01, 30, log=True),
        reg_alpha=t.suggest_float("reg_alpha", 0.001, 10, log=True),
        max_bin=t.suggest_categorical("max_bin", [256, 512, 1024])),
}
t0 = time.time()
if TRIALS:  # kat verisi bir kez hazırlanır; her deneme 5 katlı CV AUC döndürür
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    data = [fold_data(tr, va) for tr, va in folds]

    def objective(trial):
        prm = SPACE[model_name](trial) | OVR
        oof = np.zeros(n)
        for (tr, va), (Xtr, ytr, Xva, _) in zip(folds, data):
            oof[va] = MODELS[model_name](Xtr, ytr, Xva, y[va], Xva, **prm)[0]
        auc = roc_auc_score(y, oof)
        print(f"deneme {trial.number}: {auc:.5f}  {prm}  [{time.time() - t0:.0f} sn]", flush=True)
        return auc

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective, n_trials=TRIALS)
    print(f"en iyi {study.best_value:.5f}: " + ",".join(
        f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
        for k, v in study.best_params.items()), flush=True)
    sys.exit()

tag = (f"{model_name}_{'+'.join(sorted(groups))}_{LR}"
       + (f"_{NAME}" if NAME else "".join(f"_{k}={v}" for k, v in sorted(OVR.items())))
       + (f"_s{MSEED}" if MSEED != SEED else "") + (f"_f{FOLDS}" if FOLDS != 5 else "")
       + (f"_pl{PL}" if PL else "") + (f"_h{HOLD}" if HOLD >= 0 else "") + ("_mg" if MARGIN else "")
       + ("_gpu" if GPU else ""))
if FULLNIT:  # etiket CV koşusununkiyle aynı, sonuna _full
    Xa, ya, _, Xta = fold_data(np.arange(n), None)
    _, pf, _ = MODELS[model_name](Xa, ya, None, None, Xta, nit=int(FULLNIT), **OVR)
    np.save(f"pred_{tag}_full.npy", pf)
    print(f"{tag}_full: {int(FULLNIT)} ağaç, tüm train [{time.time() - t0:.0f} sn]", flush=True)
    sys.exit()
oof, pred, oof1, its = np.zeros(n), np.zeros(m), np.zeros(n), []
if MARGIN:  # katlar GLR not defteriyle aynı olmalı (10 kat, tohum 42)
    MG = np.load("glr_margins.npz")
    assert FOLDS == 10 and not use_orig and HOLD < 0 and np.isfinite(MG["oof"]).all()
for i, (tr, va) in enumerate(folds):
    Xtr, ytr, Xva, Xte = fold_data(tr, va)
    mg = {"mg": (MG["fit"][i][tr], MG["oof"][va], MG["test"][i])} if MARGIN else {}
    if MARGIN:
        assert np.isfinite(mg["mg"][0]).all()  # eğitim satırlarının hepsinin iç-kat logit'i olmalı
    oof[va], p, it = MODELS[model_name](Xtr, ytr, Xva, y[va], Xte, **OVR, **mg)
    if PL:  # sözde etiket: bu katın etiketlerini görmemiş birinci aşamanın test olasılıkları, sızıntısız
        oof1[va] = oof[va]
        Xpl = pd.concat([Xtr, Xte, Xte], ignore_index=True)
        ypl = np.r_[ytr, np.ones(m), np.zeros(m)].astype(int)
        wpl = np.r_[np.ones(len(ytr)), PL * p, PL * (1 - p)]
        oof[va], p, it = MODELS[model_name](Xpl, ypl, Xva, y[va], Xte, w=wpl, **OVR)
        print(f"  1. aşama {roc_auc_score(y[va], oof1[va]):.5f} ->", end="")
    pred += p / FOLDS
    its.append(it)
    print(f"  fold {i}: {roc_auc_score(y[va], oof[va]):.5f} ({it} it)", flush=True)
if PL:
    print(f"1. aşama (sözde etiketsiz) CV AUC: {roc_auc_score(y, oof1):.5f}", flush=True)
print(f"{tag} CV AUC: {roc_auc_score(y, oof):.5f}  "
      f"[{time.time() - t0:.0f} sn, {Xtr.shape[1]} özellik]", flush=True)
np.save(f"oof_{tag}.npy", oof)
np.save(f"pred_{tag}.npy", pred)
if FULL and HOLD >= 0:  # benzetim: sahte testte kat ortalaması / tam veri / sıra karışımları
    fs = [0.9, 1.0, 1.1, 1.2, 1.3]
    Xa, ya, _, Xta = fold_data(np.arange(n), None)
    _, pfs, _ = MODELS[model_name](Xa, ya, None, None, Xta, nit=[int(round(np.mean(its) * f)) for f in fs], **OVR)
    np.save(f"pred_{tag}_full.npy", pfs)
    rk = lambda a: pd.Series(a).rank(pct=True).to_numpy()
    pf = pfs[:, fs.index(FULL)] if FULL in fs else pfs[:, 2]
    print(f"sahte test ({len(y_hold)} satır) kat ortalaması {roc_auc_score(y_hold, pred):.6f} | tam "
          + " ".join(f"x{f}: {roc_auc_score(y_hold, pfs[:, j]):.6f}" for j, f in enumerate(fs))
          + f" | karışım (tam x{FULL} payı) " + " ".join(
              f"{w}: {roc_auc_score(y_hold, (1 - w) * rk(pred) + w * rk(pf)):.6f}" for w in (0.25, 0.5, 0.75)),
          flush=True)
elif FULL:  # tam veri: test tahmini tüm train'le eğitilmiş tek modelden (OOF yok; ağırlıklar yukarıdaki OOF'tan)
    nit = int(round(np.mean(its) * FULL))
    Xa, ya, _, Xta = fold_data(np.arange(n), None)
    _, pf, _ = MODELS[model_name](Xa, ya, None, None, Xta, nit=nit, **OVR)
    np.save(f"pred_{tag}_full.npy", pf)
    print(f"{tag}_full: {nit} ağaç (kat ortalaması {np.mean(its):.0f} x {FULL}); CV test tahminiyle sıra korr. "
          f"{np.corrcoef(pd.Series(pf).rank(), pd.Series(pred).rank())[0, 1]:.5f}", flush=True)
