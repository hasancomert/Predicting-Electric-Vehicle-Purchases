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
python blend.py pevpsubmission5 <etiket>,<etiket> ...    # virgülle verilenler (tohumlar) önce ortalanır
python train.py xgb base,freq,dig,te1 0.02 max_depth=7,subsample=0.9   # model ayarları (etikete eklenir)
python train.py xgb base,freq,dig,te1 0.02 seed=7   # başka model/hedef kodlama tohumu, katlar aynı
python train.py xgb base,freq,dig,te1 0.1 trials=40  # Optuna araması, en iyi ayarları basar
```
Uzun ayar listesi yerine `name=ad` etikete kısa bir ad koyar.

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
yeni bir sürümünü başlatır. XGBoost GPU'da CPU ile aynı skoru verir (base 0.1: 0.94193 / 0.94192, 17 / 85 sn).
CatBoost GPU'da biraz zayıf (base 0.1, aynı katlarda CPU'dan ~0.00026 düşük; GPU varsayılanı `border_count=128`
ile ~0.0008 düşük) ve 254 sınırla hızlı da değil (543 sn), bu yüzden CatBoost yerelde CPU'da eğitilir.

## Özellik grupları (`train.py`)
- `base`: ham 13 sütun
- `freq`: yaş/gelir/mesafe değerinin veride kaç kez geçtiği (train+test+orijinal)
- `dig`: gelirin `%10`, `%100`, `%1000`'i; mesafenin ondalık hanesi (sentetik veri izleri)
- `te1`: her sütunun hedef kodlaması (sklearn `TargetEncoder`, kat içinde, cross-fitting)
- `te2` / `te2s`: bütün / düşük kardinaliteli sütun ikilileri; `ted`: hane izlerinin hedef kodlaması
- `ncat`: küçük tam sayı sütunları kategori olarak; `orig`: orijinal veri ek eğitim satırı
- `bins`: gelir `/100`, `/1000` ve mesafe tam km, kaba ölçekte hedef kodlama anahtarları;
  `bins2`: ek ölçekler (gelir `/10`, `/500`, `/5000`, mesafe `/5`)
- `te3`: hedef kodlama üç yumuşatmayla (`auto`, 10, 100); `dig2`: sayısal sütunların tek tek haneleri
- `recipe`: orijinal verinin üretim formüllerinin gürültüsüz kısmı (alım ve menzil kaygısı skoru,
  C. Deotte'nin EDA'sı); `omean`: orijinal veride değer başına alım oranı
- `mb`: lgbm/xgb için `max_bin=1024` (özellik değil, model ayarı)
- `tedig`: hanelerin hedef kodlaması; `freq2`: tüm sütunların ve hanelerin normalize frekansı;
  `flag`: gelir eşik bölgeleri (30k, >=170537, 38-42k) ve çevre ilgisi 1

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

Yeni gruplar (lr 0.1, 5 kat CV AUC; XGBoost Kaggle T4 GPU'sunda). Fikirler açık not defterlerinden
(najiama "Pure LGBM", evgendvorkin "Single XGB") ve C. Deotte'nin EDA'sından:

| Gruplar (taban: base,freq,dig,te1) | LightGBM | XGBoost |
|---|---|---|
| taban | 0.94543 | 0.94551 |
| taban,recipe | 0.94544 | 0.94554 |
| taban,omean | 0.94543 | 0.94555 |
| taban,te3 | 0.94541 | 0.94555 |
| taban,dig2 | 0.94548 | 0.94561 |
| taban,mb | 0.94544 | 0.94553 |
| **taban,bins** | **0.94570** | **0.94582** |
| taban,bins,dig2 | 0.94574 | 0.94582 |
| taban,bins,te3 | 0.94574 | 0.94588 |
| taban,bins,bins2 | 0.94570 | 0.94586 |
| taban,bins,mb | 0.94570 | 0.94583 |
| **taban,bins,dig2,te3** | 0.94573 | **0.94591** |

Ayar araması (`trials=50`, XGBoost, taban,bins,dig2,te3, lr 0.1, Kaggle T4, 35 dk): varsayılan ayarlar
0.94591, en iyi 0.94606 (`max_depth=4,min_child_weight=1.616,subsample=0.914,colsample_bytree=0.5086,`
`reg_lambda=0.6968,reg_alpha=0.009999,max_bin=512`). En iyi denemelerin çoğu derinlik 4, yüksek subsample,
`max_bin` 512-1024; tepe düz (derinlik 4'lüler 0.94599-0.94606). Bu ayarlar lr 0.02'de `name=t1` ile
kullanıldı: tohum 42/7/11 0.94612/0.94613/0.94611, ortalaması 0.94619 (ayarsız 0.94603).
CatBoost (lr 0.05): taban,bins tohum 42/7 0.94603/0.94602 (ortalaması 0.94608); taban,bins,dig2,te3 tohum
42/7 0.94609/0.94609 (27-29 dk).

`t1` XGBoost lr 0.1'de taban,bins,dig2,te3 üstüne: 0.94605; +tedig 0.94601, +freq2 0.94602, +flag 0.94604,
üçü birden 0.94600 (katkı yok). LightGBM açık not defterinin ayarlarıyla (`max_depth=5,num_leaves=32,`
`min_child_samples=10,subsample=0.812763,colsample_bytree=0.30293,reg_alpha=0.07094,reg_lambda=2.03303,`
`max_bin=1024`, `name=pubp`) aynı setle lr 0.02: 0.94608 (bizim ayarlarla 0.94593, 9 dk yerel CPU).

Sinir ağı (`nn`, Kaggle T4, lr 0.002, aynı set): 12 epoch 0.94525, 30 epoch 0.94520; `lowcat=100` gömmeleri
aşırı öğreniyor (0.94228, geniş 0.94205). XGBoost ile sıra korelasyonu 0.993 (CatBoost 0.997) ama harmana
katkısı yok.

### Açık not defterleri (`public/`)
En iyi açık not defterlerinin uyarlamaları; kendi Kaggle not defterlerinde koşar, `oof_/pred_pub_*.npy`
yazar ve `blend.py` ile harmanlanır (yazarlar dosya başında):
- `pub_lgbm.py` (najiama, LightGBM, 5 kat, katlar `train.py` ile aynı): CV 0.94609 (Kaggle CPU, ~20 dk)
- `pub_xgb.py` (evgendvorkin, XGBoost, 10 kat): CV 0.94607 (Kaggle T4, ~15 dk)
Kaggle'da çalıştırmak için dosyayı `kernel-metadata.json` (`enable_gpu`, yarışma ve orijinal veri kaynakları)
ile bir klasöre koyup `kaggle kernels push -p <klasör>`, bitince
`kaggle kernels output <kullanıcı>/<slug> -p <klasör>`.

## Gönderimler
Dosya adı `pevpsubmissionN.csv`.
| No | İçerik | CV AUC | Public LB |
|---|---|---|---|
| 1 | LightGBM 0.6 + CatBoost 0.4, ham özellikler | 0.94201 | 0.94176 |
| 2 | XGBoost 0.5 + CatBoost 0.5, base,freq,dig,te1 | 0.94575 | 0.94587 |
| 3 | XGBoost 0.5 (+bins,dig2,te3) + CatBoost 0.5 (+bins) | 0.94612 | 0.94625 |
| 4 | XGBoost `t1` 3 tohum 0.75 + CatBoost (+bins,dig2,te3) 0.25 | 0.94621 | 0.94637 |
| 5 | XGBoost `t1` 3 tohum, CatBoost 2 tohum, `pub_lgbm`, `pub_xgb` (her biri 0.25) | 0.94626 | 0.94637 |

1. gönderimin tek modelleri: LightGBM 0.94190 (~3 dk), CatBoost 0.94178 (~22 dk, 4 çekirdek CPU).
2. gönderimin tek modelleri (base,freq,dig,te1): LightGBM lr 0.02 0.94562 (5.5 dk),
XGBoost lr 0.02 0.94566 (7 dk), CatBoost lr 0.05 0.94568 (24 dk). Hill climbing LightGBM'i almadı.
3. gönderimin tek modelleri: XGBoost lr 0.02 base,freq,dig,dig2,te1,te3,bins 0.94603 (Kaggle T4, 3 dk),
LightGBM aynı gruplarla lr 0.02 0.94593 (Kaggle CPU, 13 dk), CatBoost lr 0.05 base,freq,dig,te1,bins 0.94603
(yerel CPU, 24 dk). Hill climbing yine LightGBM'i almadı (XGBoost + LightGBM harmanı 0.94605).
4. gönderim: ayarlanmış XGBoost (`t1`, lr 0.02, taban,bins,dig2,te3) tohum 42/7/11 her biri 0.25 ve CatBoost
lr 0.05 taban,bins,dig2,te3 0.25. Aday olan LightGBM, ayarsız XGBoost ve iki eski CatBoost ağırlık almadı.
5. gönderim: tohumlar önce ortalanıp (blend.py'de virgülle) dört bileşen eşit ağırlıkla: XGBoost `t1` 0.94619,
CatBoost taban,bins,dig2,te3 2 tohum 0.94614, `pub_lgbm` 0.94609, `pub_xgb` 0.94607. Sinir ağı ve LightGBM
`pubp` ağırlık almadı.
