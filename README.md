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
- `tok`: gelirin GPT-2 BPE parçaları (`" " + gelir` GPT-2 ile parçalanır: ilk parça, ilk iki parça, son
  parça; hedef kodlama anahtarı + etiketsiz sayım + parça sayısı). Veri üreticisi GPT-2 tabanlı olabilir
  (GReaT türü; sayıları parça parça yazar), bu yüzden alım oranı onluk hanelerle değil parça sınırlarıyla
  değişiyor. Fikir: P. B. Elefante, "KPS6E09 Generator-Aware Ridge Logistic Regression" (tabloda 2.).
  Eşleme `gpt2_income_tokens.csv` (Kaggle'da `pvep-gpt2tok` not defteri üretir, `kaggle_run.py` onu girdi
  olarak bağlar). `tokx`: parça x bağlam hücreleri; `mix`: aynı gelir/mesafe/parçayı paylaşan train+test
  satırlarının diğer sütun ortalamaları (kendisi hariç); `chain`: gelirde ilk parça -> ilk iki parça -> tam
  değer, mesafede tam km -> değer hiyerarşik hedef oranı (katlar içinde çapraz)

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

### Açık OOF kütüphaneleri (`ext/`, `blend_ext.py`)
Başkalarının modellerinin OOF ve test tahminleri; indirme komutları `blend_ext.py`'nin başında. `blend_ext.py`
kendi modellerimizle birlikte hepsini sıra uzayında hill climbing ile harmanlar ve ağırlıkları iç içe CV ile
ölçer (4 katta seçilir, görmediği 5. katta değerlendirilir).
- `megayak/s6e9-six-feature-views-oof-library`: altı ayrı özellik hattı (A-F, 0.94608-0.94628) ve RealMLP (G,
  3 tohum 0.94618); 10 katlı sabit bölme, kat numaraları ve hedef dosyada, hedef istatistikleri kat içinde.
- `najiama/s6e9-oof`: Pure LGBM V1/V3/V5/V6, Sergey LGBM, üçlü TE XGBoost 5/10 kat (0.94533-0.94624). Bizim
  yeniden çalıştırdığımız V3 ile korelasyonu 0.9984 (0.94609 / 0.94606).

| Harman | İç içe CV | Tüm veride |
|---|---|---|
| 5. gönderim (bizim dört bileşen) | 0.94626 | 0.94626 |
| yalnızca megayak A-G | 0.94638 | 0.94640 |
| hepsi (sinir ağımız hariç) | **0.94640** | **0.94641** |

"Hepsi" seçimi: bizim CatBoost 2 tohum, naji üçlü TE XGBoost 10 kat, megayak A, B, D (her biri 0.143) ve
RealMLP 3 tohum (0.286). Test tahminlerinde bileşenler arası korelasyon OOF'takiyle en çok 0.0012 farklı.
Lojistik regresyonla istifleme de aynı (iç içe 0.94639-0.94640); harmanlama yöntemi tavanda.

Sonra eklenenler: RealMLP (`public/pub_realmlp.py`) kendi hesabımızda 6 tohumla (7/11/23/37/42/51, 5 kat, Kaggle
T4, tohum başına ~5 dk): 0.94594-0.94604, ortalaması 0.94619. CatBoost derinlik 5 (yerel CPU, 28 dk) 0.94611,
derinlik 8 (GPU) 0.94592. Son seçim (6. gönderim): RealMLP 6 tohum 0.286, CatBoost derinlik 5, naji üçlü TE
XGBoost 10 kat, megayak A, B, D (her biri 0.143); CV 0.94642, iç içe 0.94641.

10 katlı yeniden eğitim (`train.py ... folds=10`, `PVEP_FOLDS=10`; tohum 42'de bölme megayak kütüphanesiyle aynı),
her modelde ~+0.0001: RealMLP 6 tohum 0.94611-0.94614 (ortalaması 0.94624; 5 katta 0.94619), XGBoost `t1` tohum
42/7/11 0.94625/0.94623/0.94625 (ortalaması 0.94631, Kaggle T4 ~7 dk), CatBoost d5 0.94622 (yerel CPU 79 dk).
7. gönderim: RealMLP 10 kat 0.333, XGBoost 10 kat, CatBoost d5 10 kat, megayak B, D (her biri 0.167); CV 0.94644,
iç içe 0.94643.

### GPT-2 parçaları (`tok`): bulunan desen
LightGBM lr 0.1, 5 kat, taban,bins,dig2,te3 üstüne (Kaggle CPU): temel 0.94573, **+tok 0.94608**, +chain
0.94605, +tok,chain 0.94605, +tok,mix 0.94597, +mix 0.94591, +tok,tokx 0.94591. 10 katta (lr 0.02):
XGBoost `t1` tohum 42/7/11 0.94625/0.94623/0.94625 -> **0.94648/0.94645/0.94647** (ortalaması 0.94652),
RealMLP (`PVEP_TOK=1`) 0.9461 -> **0.94636** (5 tohum 0.94636-0.94638, ortalaması 0.94647), CatBoost d5
0.94622 -> **0.94645**, LightGBM `pubp` 10 kat 0.94648/0.94645 (tohum 42/7). `chain` 10 katta küçük katkı
yapıyor: XGBoost tok+chain 0.94652/0.94648 (ortalaması 0.94655), LightGBM tok+chain 0.94650; tok+mix 0.94644.
P. B. Elefante'nin GLR not defteri (üreticiye duyarlı ridge lojistik regresyon, `pvep-pbe-glr`) bizim
hesabımızda 10 katta 0.94640; ağaçlarla sıra korelasyonu 0.9965, harmanda en büyük ağırlığı alıyor.
Kat bölme tohumu 7/11 ile 0.94639/0.94638, üçünün ortalaması 0.94643. L2 düzenlileştirmesi düz: 3 / 10 / 30 ->
0.94639 / 0.94640 / 0.94640. Aynı özellik matrisiyle XGBoost
(`pvep-pbe-xgb`) 0.94636. XGBoost ayar araması tok'lu setle (40 deneme, lr 0.1): en iyi 0.94629, `t2` ayarları
(derinlik 4, `max_bin` 1024, daha az L2) 10 katta tok+chain 0.94650/0.94646, `t1` ile aynı düzey.
20 kat (10 yerine): XGBoost tok+chain 0.94651/0.94653 (ortalaması 0.94656, 10 katta 0.94656 3 tohumla),
GLR 0.94640 -> 0.94644, RealMLP+tok tek tohum 0.94637 -> 0.94641.
GLR'nin doğrusal logit'i üstüne artık MLP (`pvep-pbe-mlp`, 256-64, 2 tohum): 0.94642 (3 kat tohumu ortalaması
0.94646, 20 kat 0.94647); aynı özellik matrisiyle RealMLP 0.94638. CatBoost tok+chain 0.94648, LightGBM
tok+chain 20 kat 0.94650 (10 katla aynı). Harman doydu: tepe tırmanma ve lojistik regresyon istiflemesi
(`STACK=1`, probit(sıra)) iç içe CV'de 0.94668-0.94669; 18 güçlü modelin neredeyse eşit ortalaması en iyiler kadar iyi.
goodpjw2008'in "LR-Margin GBDT + OOF Stack" not defteri (Public LB 0.94675): GLR logit'i `init_score`/`base_margin`
olarak verilen artık LightGBM/XGBoost 5 katta 0.94646 (bizimle korelasyon 0.9968); heuljax XGB Sample 0.94631
(0.9918, en farklı üye). Sınır kuralları trende kusursuz: gelir >= 170537 393/393 alım; 31004-41970 0/1257,
mesafe >= 83 km 0/186, gelir 30000 + sübvansiyonsuz + (ilgi 1 ya da kaygı orta/yüksek) 0/7157 (`RULES=1`, OOF +0.000001).
Aynı artık yöntemi bizim tok+chain özelliklerimizle (`train.py ... margin=1`, GLR logit'leri `pvep-glr-margin` not
defterinde bizim 10 katımızla, eğitim satırları iç 5 katla): XGBoost 0.94652/0.94648 -> 0.94658/0.94654, LightGBM
0.94650 -> 0.94656 (her modelde ~+0.00006); harmana katkısı iç içe CV'de +0.00001.
Ek tohumlar: artık XGBoost s11/s23 0.94654/0.94654, `t2` 0.94656 (4 tohum ortalaması 0.94659), LightGBM s7 0.94654;
artık CatBoost (`baseline`) 0.94648 -> 0.94655. RealMLP'ye GLR logit'ini özellik olarak vermek (`PVEP_MARGIN=1`) zarar
verdi: 0.94636 -> 0.94596 (eğitim satırlarının iç-kat logit'i ile doğrulama satırlarının dış-kat logit'i farklı
dağılımda; ağaçlar bunu başlangıç payı olarak tolere ediyor, sinir ağı etmiyor). Harman 0.94671'de doydu.
20 kat (`pvep-glr-margin-f20`, GLR logit AUC 0.94644): artık XGBoost 0.94659/0.94657 (10 katta 0.94658/0.94654),
artık LightGBM 0.94659 (10 katta 0.94656). Harmana eklenince iç içe CV 0.94672 (LightGBM de eklenince 0.94671; gürültü).

### Tam veri eğitimi benzetimi (sonuç: kazanç yok)
`train.py ... hold=h,folds=9,full=1.1`: train'in %10'u etiketli sahte test, kalan %90 ile 9 katlı CV ve tam
veri modeli. XGBoost `t1` lr 0.02, 4 sahte test: tam veri modeli tek başına kat ortalamasından ortalama
0.00004 kötü (üçünde -0.00005/-0.00007, birinde +0.00003; ağaç sayısı x0.9-1.3 fark etmiyor), sıra
uzayında %25 karışım +0.00001. Tek model, 9 modelin ortalamasının varyans avantajını %11 fazla veriyle
kapatamıyor; `FULLMIX` kullanılmadı.

### Yeni desen arayışı (sonuç: bulunamadı)
7. harmanın OOF'u izotonik kalibre edilip artıkları (y - p) gruplara göre incelendi; bir grupta artık
ortalaması şanstan büyükse orada kaçırılan bir desen vardır.
- `id` sırası: hedefle AUC 0.49999; 20 binlik bloklarda alım oranı rastgele dalgalanma sınırında.
- Birebir tekrar eden satır: 955 bin satırda hiç yok.
- En yakın orijinal satırın etiketi (orijinal gürültü çekilişini taşıyabilir diye): her mesafe diliminde
  etiket uyumu modelin beklentisiyle aynı (ör. 0.798 / 0.798); sentetik satırlar orijinallerin kopyası değil.
- ~300 gruplama (tüm sütun çiftleri ve üçlüleri, gelir/mesafe/yaş ızgaraları, hane x hane ve hane x
  kategori, satın alma ve kaygı formülü tutarlılığı): en yüksek z 2.47, 225 testte şans düzeyi (~2.8).
- Sütun birleşimini yasaklamak (`solo=1`, G. Mamarin'in açık analizinden): ham sütunlarda +0.00045 ama bizim
  setimizde LightGBM lr 0.1 0.94576 -> 0.94554; harmana katkısı yok. `max_bin=15000` +0.00003.
- RealMLP'ye gelir haneleri ve ham sütunların hedef kodlaması (`PVEP_PLUS=1`): 10 kat 0.94612 (değişmedi).
- İkinci seviye: LightGBM istifleme 0.94575 (tahminleri kutulayıp sıralamayı kaybediyor); harmanı
  `init_score` verip ham sütunlar + bileşen tahminleriyle düzeltme öğrenen model 1-8 ağaçta durdu, 0.94644
  değişmedi. Yani ham sütunların hiçbir kombinasyonunda harmanın ötesinde öğrenilebilir yapı yok.
- Tablonun tepesinin büyük kısmı (0.94657'de 98 takım) OOF'suz açık harman dosyalarının tekrar gönderilmesi.
- Ardışık id'lerde etiket/özellik korelasyonu yok (gecikme 1: +0.001, şans sınırı ±0.0012); train ve test
  ayırt edilemiyor (düşmanca doğrulama AUC 0.50005).
- Başka 25 açık OOF modeli (golem 19 model, hirge sinir ağları: FT-Transformer/TabM/MoE, digit-leak, BlamerX,
  residual-stack) tek tek 8. harmana eklendi: yalnızca residual-stack v19 (+0.000014), BlamerX (+0.000007) ve
  jazivxt zoom-zoom (+0.000006) katkı yaptı; üçü `blend_ext.py`'de aday. "marcmaldonado/s6e9-generator-
  fingerprints" üreticinin orijinal gelir değerlerini kopyaladığını (%97,9) gösteriyor, ama yazarın ölçümüne
  göre orijinal satırın etiketi tam değer kodlamasının üstüne +0.00003 ekliyor (bizim omean bulgumuzla aynı).
- Sözde etiketleme (`train.py ... pl=w`, kat başına sızıntısız): ham sütunlarda 0.94164 -> 0.94172, ama
  XGBoost `t1` 10 katta 0.94625 -> 0.94626 (w=1 ve 0.5); harmana katkısı yok.
- Saf aralık kuralları (trende %0/%100 alım olan gelir/mesafe aralıklarını en alta/üste itmek): 4 katta bulunup
  görülmemiş katta uygulanınca >=100 satırlık kurallar -0.00037, >=300 satır -0.00004; yalnızca büyük ölü bölge
  (gelir 38174-41384, trende 1257 satır, 0 alım) +0.000004 (`DEADZONE=1`). Public LB'de 0.94657 alan açık harman
  (OOF'suz) bizden en çok bu tür bölgelerde ayrılıyor (ölü bölgede ortalama sıra farkı -0.12); iki dosyanın
  public LB farkı (0.00012) bu kadar benzer dosyalar için beklenen gürültü düzeyinde (~0.0001), karıştırılmadı.

### Açık not defterleri (`public/`)
En iyi açık not defterlerinin uyarlamaları; kendi Kaggle not defterlerinde koşar, `oof_/pred_pub_*.npy`
yazar ve `blend.py` ile harmanlanır (yazarlar dosya başında):
- `pub_lgbm.py` (najiama, LightGBM, 5 kat, katlar `train.py` ile aynı): CV 0.94609 (Kaggle CPU, ~20 dk)
- `pub_xgb.py` (evgendvorkin, XGBoost, 10 kat): CV 0.94607 (Kaggle T4, ~15 dk)
- `pub_realmlp.py` (yekenot, RealMLP, PyTorch, `PVEP_SEED` tohumu, `PVEP_FOLDS`, `PVEP_PLUS`): 5 katta tohum başına
  ~0.9460 (Kaggle T4, ~5 dk), 10 katta ~0.9461 (~11 dk)
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
| 6 | sıra uzayında, açık OOF kütüphaneleri + RealMLP 6 tohum (`blend_ext.py`, aşağıda) | 0.94642 (iç içe 0.94641) | 0.94642 |
| 7 | 6. gönderimin yapısı, kendi modellerimiz 10 katla (`folds=10`) | 0.94644 (iç içe 0.94643) | 0.94644 |
| 8 | 7. + RealMLP `PVEP_PLUS` 2 tohum (desen arayışının artığı) | 0.94645 (iç içe 0.94644) | 0.94644 |
| 9 | 8. + üç açık OOF (residual-stack v19 ve jazivxt, BlamerX) | 0.94647 (iç içe 0.94645) | 0.94645 |
| 10 | yalnız kendi modellerimiz: RealMLP 10 kat 0.5 + XGBoost `t1` 10 kat 0.5, ölü bölge kuralı | 0.94642 (iç içe 0.94642) | 0.94644 |
| 11 | GPT-2 parçaları (`tok`): XGBoost `t1`+tok 10 kat 3 tohum 0.5, RealMLP+tok 0.125, RealMLP 10 kat 0.125, residual-stack v19 0.125, BlamerX 0.125, ölü bölge | 0.94661 (iç içe 0.94660) | **0.94667** |
| 12 | tok'lu modeller: CatBoost d5+tok, LightGBM tok+chain, XGBoost tok+chain 2 tohum, RealMLP+tok 5 tohum, residual-stack v19 (her biri 0.143), P. B. Elefante GLR (0.94640; 0.286), ölü bölge | 0.94667 (iç içe 0.94666) | **0.94671** |
| 13 | 12. + XGBoost tok+chain 3 tohum ve `t2` ayarları, GLR 3 kat tohumu (42/7/11) ortalaması, GLR özellikleriyle XGBoost (0.94636), RealMLP+tok 6 tohum; ağırlıklar XGB tok+chain / RealMLP+tok / GLR 0.2, GLR-XGB, LightGBM tok+chain, megayak D, residual-stack v19 0.1 | 0.94669 (iç içe 0.94667) || **0.94671** |
| 14 | 13. + 20 kat (her model verinin %95'i): XGBoost tok+chain 20 kat 2 tohum 0.2 (10 kat yerine), GLR 20 kat 0.1 (+ 3 tohumlu 10 kat 0.1) | 0.94669 (iç içe 0.94668) || **0.94671** |
| 15 | tok'lu 18 model (XGBoost/LightGBM/CatBoost/RealMLP 10 ve 20 kat, GLR, GLR üstüne MLP, GLR özellikli XGBoost ve RealMLP, megayak D, residual-stack v19/jaz, BlamerX) probit(sıra) üzerinde lojistik regresyonla istiflendi (`STACK=1`; katsayılar 0.18-0.20, neredeyse eşit ortalama), ölü bölge | 0.94669 (iç içe 0.94669) || **0.94671** |
| 16 | 15.'in adayları + goodpjw2008'in GLR logit'inden başlayan artık LightGBM/XGBoost (0.94646), heuljax XGB Sample (0.94631), BlamerX pencere XGBoost; tepe tırmanma (XGB tok+chain 20 kat 0.2, GLR-MLP 20 kat 0.2, artık LightGBM, heuljax XGB, GLR-XGB, LightGBM tok+chain, RealMLP+tok, residual-stack v19 0.1); dört sınır kuralı (`RULES=1`) | 0.94671 (iç içe 0.94670) || **0.94673** |
| 17 | 16. + GLR logit'inden artık XGBoost tok+chain (`margin=1`, 0.94658/0.94654; ağırlık 0.3) | 0.94672 (iç içe 0.94671) || **0.94675** |
| 18 | 17. + artık XGBoost 4 tohum ve `t2`, artık LightGBM 2 tohum, artık CatBoost d5 (0.94655); ağırlıklar iç içe 5 katta seçilip ortalandı (`AVGW=1`; artık XGB 0.24, XGB tok+chain 20 kat 0.18, artık CatBoost, heuljax XGB, GLR-XGB, residual-stack v19 ~0.1, RealMLP+tok 0.09, 5 küçük üye), dört sınır kuralı | 0.94672 (iç içe 0.94671) || **0.94676** |
| 19 | 18. + 20 katlı GLR logit'inden 20 katlı artık XGBoost (2 tohum 0.94661; ağırlık 0.28); iç içe katlarda ortalanan ağırlıklar (`AVGW=1`), dört sınır kuralı | 0.94673 (iç içe 0.94672) || **0.94676** |

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
