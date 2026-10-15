"""Kendi modellerimiz + açık OOF kütüphaneleri (ext/) üzerinde sıra uzayında hill climbing.
Ağırlıklar iç içe CV ile de ölçülür (4 katta seçilir, 5.'de değerlendirilir), aşırı öğrenme görünsün.
Kullanım: python blend_ext.py pevpsubmission6 [aday,aday,...]   (aday verilmezse hepsi, sinir ağımız hariç)
FULLMIX=w: kendi modellerimizin test tahmininde tam veri modellerinin payı (aşağıda).
DEADZONE=1: gelir 38174-41384 (trende 1257 satır, hiç alım yok; kurallar görülmemiş katta sınandı) en alta itilir.
ext/ için: kaggle datasets download -d najiama/s6e9-oof -p ext/s6e9-oof --unzip
           kaggle datasets download -d megayak/s6e9-six-feature-views-oof-library \\
               -p ext/s6e9-six-feature-views-oof-library --unzip
           (isteğe bağlı) legtarrr/s6e9-residual-stack-oof, medvax/s6e9-medvax-blamerx-oof-predictions"""
import glob
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

tr = pd.read_csv("train.csv", usecols=["id", "Will_Buy_EV"])
te = pd.read_csv("sample_submission.csv")
y = (tr.Will_Buy_EV == "Yes").to_numpy().astype(int)
rank = lambda v: rankdata(v) / len(v)
O, P = {}, {}  # aday -> OOF / test sıraları

# kendi modellerimiz (virgülle verilen tohumlar ortalanır)
X = "xgb_base+bins+dig+dig2+freq+te1+te3_0.02_t1"
C = "cat_base+bins+dig+dig2+freq+te1+te3_0.05"
OWN = {"our_xgb3": f"{X}_gpu,{X}_s7_gpu,{X}_s11_gpu", "our_cat2": f"{C},{C}_s7",
       "our_pub_lgbm": "pub_lgbm", "our_pub_xgb": "pub_xgb",
       "our_lgbm_pubp": "lgbm_base+bins+dig+dig2+freq+te1+te3_0.02_pubp",
       "our_nn": "nn_base+bins+dig+dig2+freq+te1+te3_0.002_gpu"}
RM = sorted(glob.glob("oof_pub_realmlp_s*.npy"))  # RealMLP tohumları (public/pub_realmlp.py)
if RM:
    OWN["our_realmlp"] = ",".join(f[4:-4] for f in RM)
RM10 = sorted(glob.glob("oof_pub_realmlp_f10_s*.npy"))  # 10 katlı RealMLP tohumları
if RM10:
    OWN["our_realmlp_f10"] = ",".join(f[4:-4] for f in RM10)
RMP = sorted(glob.glob("oof_pub_realmlp_plus_f10_s*.npy"))  # PVEP_PLUS=1 ile RealMLP (haneler + ham TE)
if RMP:
    OWN["our_realmlp_plus_f10"] = ",".join(f[4:-4] for f in RMP)
XF10 = [t for t in (f"{X}_f10_gpu", f"{X}_s7_f10_gpu", f"{X}_s11_f10_gpu") if os.path.exists(f"oof_{t}.npy")]
if XF10:
    OWN["our_xgb_f10"] = ",".join(XF10)
CF10 = [t for t in (f"{C}_d5_f10", f"{C}_d5_s7_f10") if os.path.exists(f"oof_{t}.npy")]  # CatBoost d5 10 kat
if CF10:
    OWN["our_cat_d5_f10"] = ",".join(CF10)
# GPT-2 BPE gelir parçalarıyla (tok) eğitilenler
XT, CT = "xgb_base+bins+dig+dig2+freq+te1+te3+tok_0.02_t1", "cat_base+bins+dig+dig2+freq+te1+te3+tok_0.05_d5"
for k, tags in {"our_xgb_tok_f10": [f"{XT}_f10_gpu", f"{XT}_s7_f10_gpu", f"{XT}_s11_f10_gpu"],
                "our_cat_tok_f10": [f"{CT}_f10", f"{CT}_s7_f10"],
                "our_cat_tokchain_f10": ["cat_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.05_d5_f10"],
                "our_lgbm_tok_f10": ["lgbm_base+bins+dig+dig2+freq+te1+te3+tok_0.02_pubp_f10",
                                     "lgbm_base+bins+dig+dig2+freq+te1+te3+tok_0.02_pubp_s7_f10"],
                "our_lgbm_tokchain_f10": ["lgbm_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.02_pubp_f10",
                                          "lgbm_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.02_pubp_s7_f10"],
                "our_xgb_tokchain_f10": [f"xgb_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.02_t1{s}_f10_gpu"
                                         for s in ("", "_s7", "_s11")],
                "our_xgb_t2_tokchain_f10": [f"xgb_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.02_t2{s}_f10_gpu"
                                            for s in ("", "_s7")],
                "our_lgbm_tokmix_f10": ["lgbm_base+bins+dig+dig2+freq+mix+te1+te3+tok_0.02_pubp_f10"],
                "our_realmlp_tok_f10": [f[4:-4] for f in sorted(glob.glob("oof_pub_realmlp_tok_f10_s*.npy"))],
                # 20 kat: her model verinin %95'iyle
                "our_xgb_tokchain_f20": [f"xgb_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.02_t1{s}_f20_gpu"
                                         for s in ("", "_s7")],
                "our_lgbm_tokchain_f20": ["lgbm_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.02_pubp_f20"],
                # GLR logit'inden başlayan artık modeller (margin=1)
                "our_xgb_tokchain_mg": [f"xgb_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.02_t1{s}_f10_mg_gpu"
                                        for s in ("", "_s7", "_s11", "_s23")],
                "our_xgb_t2_tokchain_mg": ["xgb_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.02_t2_f10_mg_gpu"],
                "our_lgbm_tokchain_mg": [f"lgbm_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.02_pubp{s}_f10_mg"
                                         for s in ("", "_s7")],
                "our_cat_tokchain_mg": ["cat_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.05_d5_f10_mg"],
                # 20 katlı GLR logit'inden, 20 katlı artık modeller
                "our_xgb_tokchain_mg_f20": [f"xgb_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.02_t1{s}_f20_mg_gpu"
                                            for s in ("", "_s7")],
                "our_lgbm_tokchain_mg_f20": ["lgbm_base+bins+chain+dig+dig2+freq+te1+te3+tok_0.02_pubp_f20_mg"],
                "our_realmlp_tok_mg": [f[4:-4] for f in sorted(glob.glob("oof_pub_realmlp_tok_mg_f10_s*.npy"))],
                "our_realmlp_tok_f20": [f[4:-4] for f in sorted(glob.glob("oof_pub_realmlp_tok_f20_s*.npy"))]}.items():
    tags = [t for t in tags if os.path.exists(f"oof_{t}.npy")]
    if tags:
        OWN[k] = ",".join(tags)
for k, t in {"our_cat_d8": f"{C}_d8_gpu", "our_cat_d5": f"{C}_d5"}.items():
    if os.path.exists(f"oof_{t}.npy"):
        OWN[k] = t
for k, tags in OWN.items():
    load = lambda kind: np.mean([np.load(f"{kind}_{t}.npy") for t in tags.split(",")], axis=0)
    O[k], P[k] = rank(load("oof")), rank(load("pred"))


def aligned(frame, ids, col):
    return pd.DataFrame({"id": ids}).merge(frame[["id", col]], on="id", how="left",
                                           validate="one_to_one")[col].to_numpy(float)


# najiama/s6e9-oof: ayrı OOF/test dosyaları
N = "ext/s6e9-oof/"
for k, f in {"naji_lgbm_v1": "Pure LGBM_V1", "naji_lgbm_v3": "Pure LGBM_V3", "naji_lgbm_v5": "Pure LGBM_V5",
             "naji_lgbm_v6": "Pure LGBM_V6", "naji_xgb_te_5f": "XGBoost_Triple_TE_5folds",
             "naji_xgb_te_10f": "XGBoost_Triple_TE_10folds"}.items():
    fo, ft = pd.read_csv(f"{N}{f}_oof.csv"), pd.read_csv(f"{N}{f}_test.csv")
    O[k] = rank(aligned(fo, tr.id, [c for c in fo if c != "id"][-1]))
    P[k] = rank(aligned(ft, te.id, [c for c in ft if c != "id"][-1]))
fo, ft = pd.read_csv(f"{N}Sergey_LGBM_oof.csv"), pd.read_csv(f"{N}Sergey_LGBM_submission.csv")
O["naji_sergey"] = rank(aligned(fo, tr.id, [c for c in fo if c != "id"][-1]))
P["naji_sergey"] = rank(aligned(ft, te.id, [c for c in ft if c != "id"][-1]))

# megayak: altı özellik görünümü (A-F) ve RealMLP (G), 10 katlı sabit bölme, hedefler dosyada
M = "ext/s6e9-six-feature-views-oof-library/"
for fo, ft in [("oof_six_views.csv", "test_six_views.csv"), ("oof_realmlp_g.csv", "test_realmlp_g.csv")]:
    fo, ft = pd.read_csv(M + fo), pd.read_csv(M + ft)
    assert np.array_equal(aligned(fo, tr.id, "Will_Buy_EV").astype(int), y)
    for c in [c for c in ft if c not in ("id", "ensemble")]:
        k = "mv_" + (c if c.startswith("G_") else c[0])  # mv_A..mv_F, mv_G_realmlp_...
        O[k], P[k] = rank(aligned(fo, tr.id, c)), rank(aligned(ft, te.id, c))

# FULLMIX=w: kendi bileşenlerimizin test tahmini = (1-w) x 10 katlı modellerin ortalaması + w x tüm train ile eğitilmiş
# model(ler) (train.py full=/fullnit=, pub_realmlp.py PVEP_FULL=1). Ağırlıklar yine OOF'tan; yalnız test tahmini değişir.
FULLMIX = float(os.environ.get("FULLMIX", "0"))
FULLMAP = {"our_realmlp_f10": sorted(glob.glob("pred_pub_realmlp_full_s*.npy")),
           "our_xgb_f10": [f"pred_{t}_full.npy" for t in XF10 if os.path.exists(f"pred_{t}_full.npy")],
           "our_cat_d5_f10": [f"pred_{t}_full.npy" for t in CF10 if os.path.exists(f"pred_{t}_full.npy")]}
for k, files in FULLMAP.items():
    if FULLMIX and k in P and files:
        pf = rank(np.mean([np.load(f) for f in files], axis=0))
        print(f"{k}: test tahmininin %{FULLMIX * 100:.0f}'i tam veri modellerinden ({len(files)} dosya), "
              f"10 katlı ortalamayla sıra korr. {np.corrcoef(pf, P[k])[0, 1]:.5f}")
        P[k] = rank((1 - FULLMIX) * P[k] + FULLMIX * pf)

# Diğer açık OOF kütüphanelerinden harmana katkı yapan üçü (25 aday tek tek denendi, gerisi 0 ağırlık aldı):
# legtarrr/s6e9-residual-stack-oof (v19 ve jazivxt'in "zoom zoom" modeli) ve medvax/s6e9-medvax-blamerx-oof-predictions.
for k, fo, ft in [("rs_v19", "s6e9-residual-stack-oof/v19_oof.csv", "s6e9-residual-stack-oof/v19_test.csv"),
                  ("rs_jaz", "s6e9-residual-stack-oof/jaz_oof.csv", "s6e9-residual-stack-oof/jaz_test.csv"),
                  ("medvax_blamerx", "s6e9-medvax-blamerx-oof-predictions/oof.csv",
                   "s6e9-medvax-blamerx-oof-predictions/submission.csv")]:
    if os.path.exists("ext/" + fo):
        fo, ft = pd.read_csv("ext/" + fo), pd.read_csv("ext/" + ft)
        O[k], P[k] = rank(aligned(fo, tr.id, "pred")), rank(aligned(ft, te.id, [c for c in ft if c != "id"][-1]))
# P. B. Elefante'nin "Generator-Aware Ridge Logistic Regression" not defteri, bizim hesabımızda (pvep-pbe-glr)
# Kat bölme tohumları 42/7/11 (ext/pbe-glr, ext/pbe-glr-s7, ...) ortalanır: her satırın OOF'u onu görmemiş modellerden.
GS = [g for g in ("ext/pbe-glr/", "ext/pbe-glr-s7/", "ext/pbe-glr-s11/")
      if os.path.exists(g + "GENERATOR_AWARE_LOGREG_SAMPLE_OOF.parquet")]
if GS:
    rd = lambda g, kind: pd.read_parquet(g + f"GENERATOR_AWARE_LOGREG_SAMPLE_{kind}.parquet")
    O["pbe_glr"] = rank(np.mean([aligned(rd(g, "OOF"), tr.id, "oof_pred") for g in GS], axis=0))
    P["pbe_glr"] = rank(np.mean([aligned(rd(g, "TEST"), te.id, "test_pred") for g in GS], axis=0))
    print(f"pbe_glr: {len(GS)} kat tohumu")
GS20 = [g for g in ("ext/pbe-glr-f20/", "ext/pbe-glr-f20-s7/", "ext/pbe-glr-f20-s11/")  # 20 katlı GLR, kat tohumları
        if os.path.exists(g + "GENERATOR_AWARE_LOGREG_SAMPLE_OOF.parquet")]
if GS20:
    O["pbe_glr_f20"] = rank(np.mean([aligned(rd(g, "OOF"), tr.id, "oof_pred") for g in GS20], axis=0))
    P["pbe_glr_f20"] = rank(np.mean([aligned(rd(g, "TEST"), te.id, "test_pred") for g in GS20], axis=0))
    print(f"pbe_glr_f20: {len(GS20)} kat tohumu")
# GLR'nin doğrusal logit'i üstüne artık MLP (pvep-pbe-mlp)
for k, dirs in {"pbe_mlp": ("ext/pbe-mlp/", "ext/pbe-mlp-s7/", "ext/pbe-mlp-s11/"), "pbe_mlp_f20": ("ext/pbe-mlp-f20/",)}.items():
    dirs = [g for g in dirs if os.path.exists(g + "GENERATOR_AWARE_MLP_OOF.parquet")]  # kat tohumları ortalanır
    if dirs:
        rm = lambda g, kind: pd.read_parquet(g + f"GENERATOR_AWARE_MLP_{kind}.parquet")
        O[k] = rank(np.mean([aligned(rm(g, "OOF"), tr.id, "oof_pred") for g in dirs], axis=0))
        P[k] = rank(np.mean([aligned(rm(g, "TEST"), te.id, "test_pred") for g in dirs], axis=0))
        print(f"{k}: {len(dirs)} kat tohumu")
# goodpjw2008 "LR-Margin GBDT + OOF Stack" (GLR logit'inden başlayan artık LightGBM/XGBoost, 5 kat) ve heuljax XGB Sample
GP = "ext/s6e9-lr-margin-gbdt-oof-stack-lb-0-94675/"
if os.path.exists(GP + "oof_mine.csv"):
    fo, ft = pd.read_csv(GP + "oof_mine.csv"), pd.read_csv(GP + "test_mine.csv")
    for c, k in (("residual_lgbm", "gp_resid_lgbm"), ("residual_xgb", "gp_resid_xgb"), ("glm", "gp_glm5")):
        O[k], P[k] = rank(aligned(fo, tr.id, c)), rank(aligned(ft, te.id, c))
HX = "ext/kps6e09-xgb-sample/"
if os.path.exists(HX + "oof/XGB_SAMPLE_OOF.parquet"):
    fo, ft = pd.read_parquet(HX + "oof/XGB_SAMPLE_OOF.parquet"), pd.read_parquet(HX + "test_preds/XGB_SAMPLE_TEST.parquet")
    O["heuljax_xgb"], P["heuljax_xgb"] = rank(aligned(fo, tr.id, "oof_pred")), rank(aligned(ft, te.id, "test_pred"))
BW = "ext/s6e9-xgboost-window-encodings-0-946-cv/"
if os.path.exists(BW + "oof.csv"):
    fo, ft = pd.read_csv(BW + "oof.csv"), pd.read_csv(BW + "submission.csv")
    O["blamerx_win"], P["blamerx_win"] = rank(aligned(fo, tr.id, "pred")), rank(aligned(ft, te.id, "Will_Buy_EV"))
# Aynı özellik matrisiyle RealMLP (pvep-pbe-realmlp)
if os.path.exists("ext/pbe-realmlp/GENERATOR_AWARE_REALMLP_OOF.parquet"):
    fo = pd.read_parquet("ext/pbe-realmlp/GENERATOR_AWARE_REALMLP_OOF.parquet")
    ft = pd.read_parquet("ext/pbe-realmlp/GENERATOR_AWARE_REALMLP_TEST.parquet")
    O["pbe_realmlp"], P["pbe_realmlp"] = rank(aligned(fo, tr.id, "oof_pred")), rank(aligned(ft, te.id, "test_pred"))
# Aynı özellik matrisiyle lojistik regresyon yerine XGBoost (pvep-pbe-xgb)
if os.path.exists("ext/pbe-xgb/GENERATOR_AWARE_XGB_OOF.parquet"):
    fo, ft = pd.read_parquet("ext/pbe-xgb/GENERATOR_AWARE_XGB_OOF.parquet"), pd.read_parquet("ext/pbe-xgb/GENERATOR_AWARE_XGB_TEST.parquet")
    O["pbe_xgb"], P["pbe_xgb"] = rank(aligned(fo, tr.id, "oof_pred")), rank(aligned(ft, te.id, "test_pred"))
assert all(np.isfinite(v).all() for v in [*O.values(), *P.values()])


def hill(names, idx, max_steps=100):
    """Tekrar seçilebilir açgözlü ağırlıklandırma; en iyi tekten başlar."""
    A = np.column_stack([O[k][idx] for k in names])
    single = [roc_auc_score(y[idx], a) for a in A.T]
    cnt = np.zeros(len(names))
    cnt[int(np.argmax(single))] = 1
    best, s = max(single), A @ cnt
    for _ in range(max_steps):
        v, j = max((roc_auc_score(y[idx], s + A[:, j]), j) for j in range(len(names)))
        if v <= best + 1e-7:
            break
        best, cnt[j], s = v, cnt[j] + 1, s + A[:, j]
    return cnt / cnt.sum()


out = sys.argv[1]
names = sys.argv[2].split(",") if len(sys.argv) > 2 else [k for k in O if k != "our_nn"]
for k in names:
    print(f"{roc_auc_score(y, O[k]):.5f}  {k}")
blind = np.zeros(len(y))
if os.environ.get("STACK") == "1":  # lojistik regresyonla istifleme: probit(sıra) üzerinde sürekli ağırlıklar
    from scipy.stats import norm
    from sklearn.linear_model import LogisticRegression
    zf = lambda M: norm.ppf(np.clip(M, 1e-6, 1 - 1e-6))
    ZO, ZP = zf(np.column_stack([O[k] for k in names])), zf(np.column_stack([P[k] for k in names]))
    C_ = float(os.environ.get("STACK_C", "1"))
    fitw = lambda idx: LogisticRegression(C=C_, max_iter=3000).fit(ZO[idx], y[idx]).coef_[0]
    for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=42).split(y, y):
        blind[va_i] = ZO[va_i] @ fitw(tr_i)
    w = fitw(np.arange(len(y)))
    print("katsayılar:", {k: round(float(x), 3) for k, x in sorted(zip(names, w), key=lambda t: -abs(t[1]))[:15]})
    oo, pt = rank(ZO @ w), rank(ZP @ w)
else:
    ws = []
    for tr_i, va_i in StratifiedKFold(5, shuffle=True, random_state=42).split(y, y):
        ws.append(hill(names, tr_i))
        blind[va_i] = np.column_stack([O[k][va_i] for k in names]) @ ws[-1]
    # AVGW=1: iç içe katlarda seçilen 5 ağırlık vektörünün ortalaması (tek seferlik tüm-veri seçiminden daha sağlam)
    w = np.mean(ws, axis=0) if os.environ.get("AVGW") == "1" else hill(names, np.arange(len(y)))
    print("ağırlıklar:", {k: round(float(x), 3) for k, x in zip(names, w) if x})
    oo, pt = np.column_stack([O[k] for k in names]) @ w, np.column_stack([P[k] for k in names]) @ w
print(f"harman CV AUC: {roc_auc_score(y, oo):.5f}  (iç içe CV: {roc_auc_score(y, blind):.5f})")
if os.environ.get("DEADZONE") == "1":  # ölü bölge: sıralamayı koruyarak en alta
    inc_tr = pd.read_csv("train.csv", usecols=["Annual_Income_USD"]).Annual_Income_USD.to_numpy()
    inc_te = pd.read_csv("test.csv", usecols=["Annual_Income_USD"]).Annual_Income_USD.to_numpy()
    zt, zv = (inc_tr >= 38174) & (inc_tr <= 41384), (inc_te >= 38174) & (inc_te <= 41384)
    oz = np.where(zt, oo * 1e-3 - 1, oo)
    print(f"ölü bölge: train {zt.sum()} satır ({int(y[zt].sum())} alım), OOF {roc_auc_score(y, oo):.6f} -> "
          f"{roc_auc_score(y, oz):.6f}; test {zv.sum()} satır en alta")
    pt = np.where(zv, pt * 1e-3 - 1, pt)
if os.environ.get("RULES") == "1":  # dört sınır kuralı (goodpjw2008; trende %100 / %0 alım): sıralamayı koruyarak uçlara
    def rules(df):
        inc, km = df.Annual_Income_USD.to_numpy(), df.Daily_Commute_km.to_numpy()
        nosub, env1 = (df.Subsidy_Available == "No").to_numpy(), (df.Environmental_Concern_Level == 1).to_numpy()
        anx = df.Range_Anxiety_Level.isin(["Medium", "High"]).to_numpy()
        bot = ((inc >= 31004) & (inc <= 41970)) | (km >= 83) | ((inc == 30000) & nosub & (env1 | anx))
        return inc >= 170537, bot
    shift = lambda p, t, b: rank(p) + 2.0 * t - 2.0 * b
    (tt, tb), (vt, vb) = rules(pd.read_csv("train.csv")), rules(pd.read_csv("test.csv"))
    print(f"kurallar: train üst {tt.sum()} ({int(y[tt].sum())} alım), alt {tb.sum()} ({int(y[tb].sum())} alım); OOF "
          f"{roc_auc_score(y, oo):.6f} -> {roc_auc_score(y, shift(oo, tt, tb)):.6f}; test üst {vt.sum()}, alt {vb.sum()}")
    pt = shift(pt, vt, vb)
te["Will_Buy_EV"] = rank(pt)
te.to_csv(f"{out}.csv", index=False)
print(f"{out}.csv yazıldı, {len(te)} satır")
