# Kaggle: Predicting Electric Vehicle Purchases (Playground S6E9)

Hedef `Will_Buy_EV` (Yes/No), gönderim olasılık, metrik ROC AUC.

## Veri
Depo köküne (Git'e girmez):
- Yarışma zip'indeki `train.csv`, `test.csv`, `sample_submission.csv`
- `original.csv`: orijinal veri (10.000 satır), Kaggle'da `itzzomkar/ev-adoption-behavior-and-range-anxiety`
  (`EV_Adoption_and_Range_Anxiety_Dataset.csv`). Kopyası: GitHub `thasveer-art/EV-Adoption-Range-Anxiety-Analysis-using-Python`.

## Kaggle bağlantısı (GPU)
Ortamda `KAGGLE_USERNAME` + `KAGGLE_KEY` (ya da `KAGGLE_API_TOKEN`) ve ağ izni
(`www.kaggle.com`, `api.kaggle.com`, `storage.googleapis.com`; not defteri çıktısını indirmek için
`www.kaggleusercontent.com`) varsa: veri `kaggle competitions download`
ile iner, ağır eğitim `kaggle kernels push` ile Kaggle GPU'sunda çalışır. Gönderim yalnızca kullanıcı isteyince.
```
python -m pip install kaggle
kaggle competitions download -c playground-series-s6e9 -p . && unzip -o playground-series-s6e9.zip
kaggle datasets download -d itzzomkar/ev-adoption-behavior-and-range-anxiety -p . --unzip
mv EV_Adoption_and_Range_Anxiety_Dataset.csv original.csv
```

## Çalıştırma
```
python -m pip install -r requirements.txt
python train.py lgbm base,freq,dig,te1 0.02     # oof_/pred_<etiket>.npy
python train.py xgb  base,freq,dig,te1 0.02
python train.py cat  base,freq,dig,te1 0.05
python blend.py pevpsubmission3 <etiket> <etiket> ...   # hill climbing, pevpsubmission3.csv
```

### Kaggle GPU'sunda eğitim
`kaggle_run.py`, `train.py`'yi özel bir Kaggle not defterinde çalıştırır (`<kullanıcı>/pvep-train`, T4 GPU,
internet kapalı; veri yarışmadan ve orijinal veri setinden bağlanır). Her tırnaklı argüman bir `train.py`
çağrısıdır, hepsi aynı oturumda sırayla koşar. Bitince günlük (fold skorları, CV AUC) ekrana basılır,
`oof_/pred_*.npy` depo köküne iner ve `blend.py` ile yerel dosyalarla birlikte kullanılabilir.
```
python kaggle_run.py "xgb base,freq,dig,te1 0.02" "cat base,freq,dig,te1 0.05"
python kaggle_run.py --fetch    # bekleme yarıda kalırsa: son sürümü bekle, günlüğü göster, indir
kaggle quota                    # haftalık GPU kotası (30 saat)
```
xgb ve cat GPU'da eğitilir, etiketin sonuna `_gpu` eklenir (`xgb_base_0.1_gpu`); GPU sonuçları CPU'dakilerden
biraz farklı olabilir. pip'in LightGBM'i CUDA'sız olduğundan lgbm orada da CPU'da koşar. Her push not defterinin
yeni bir sürümünü başlatır.

## Özellik grupları (`train.py`)
- `base`: ham 13 sütun
- `freq`: yaş/gelir/mesafe değerinin veride kaç kez geçtiği (train+test+orijinal)
- `dig`: gelirin `%10`, `%100`, `%1000`'i; mesafenin ondalık hanesi (sentetik veri izleri)
- `te1`: her sütunun hedef kodlaması (sklearn `TargetEncoder`, kat içinde, cross-fitting)
- `te2` / `te2s`: bütün / düşük kardinaliteli sütun ikilileri; `ted`: hane izlerinin hedef kodlaması
- `ncat`: küçük tam sayı sütunları kategori olarak; `orig`: orijinal veri ek eğitim satırı

Deneyler (LightGBM, öğrenme oranı 0.1, 5 kat CV AUC):

| Gruplar | CV AUC |
|---|---|
| base | 0.94178 |
| base,orig | 0.94177 |
| base,freq | 0.94301 |
| base,dig | 0.94354 |
| base,freq,dig | 0.94381 |
| base,freq,dig,ncat | 0.94307 |
| **base,freq,dig,te1** | **0.94543** |
| base,freq,dig,te1,orig | 0.94534 |
| base,freq,dig,te1,ted | 0.94537 |
| base,freq,dig,te2 | 0.94522 |
| base,freq,dig,te2s | 0.94535 |

XGBoost aynı ayarla base,freq,dig,te1: 0.94554.

## Gönderimler
Dosya adı `pevpsubmissionN.csv`.
| No | İçerik | CV AUC | Public LB |
|---|---|---|---|
| 1 | LightGBM 0.6 + CatBoost 0.4, ham özellikler | 0.94201 | 0.94176 |
| 2 | XGBoost 0.5 + CatBoost 0.5, base,freq,dig,te1 | 0.94575 | 0.94587 |

1. gönderimin tek modelleri: LightGBM 0.94190 (~3 dk), CatBoost 0.94178 (~22 dk, 4 çekirdek CPU).
2. gönderimin tek modelleri (base,freq,dig,te1): LightGBM lr 0.02 0.94562 (5.5 dk),
XGBoost lr 0.02 0.94566 (7 dk), CatBoost lr 0.05 0.94568 (24 dk). Hill climbing LightGBM'i almadı.
