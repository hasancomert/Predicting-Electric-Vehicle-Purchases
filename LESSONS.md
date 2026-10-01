# LESSONS: Kaggle Playground S6E9'dan çıkan dersler

Bu dosya, çalışmayı başka bir ikili sınıflandırma yarışmasına taşırken başvurmak için yazıldı. Yalnızca gerçekten
çalıştırılıp ölçülmüş şeyler var; sayıların kaynağı README, kod, Kaggle gönderim listesi ve leaderboard dosyası.

- **Görev:** `Will_Buy_EV` (Yes/No) tahmini, metrik ROC AUC. Train 668.665, test 286.571 satır; orijinal veri 10.000 satır.
- **Sonuç:** son gönderim (19) public **0.94676**, private **0.94570**. Son public leaderboard'da (1 Ekim 2026'da
  indirildi) 3.576 takım arasında 133. sıra. Private sıralama API'den alınamadı, bu yüzden burada yok.

## 1. Validation kurulumu ve skor uyumu

### Kurulum

| Konu | Ne yapıldı |
|---|---|
| Bölme | `StratifiedKFold(K, shuffle=True, random_state=42)`. `seed=` model ve hedef kodlama tohumunu değiştirir, katlar hep aynı kalır. |
| Kat sayısı | Önce 5, 7. gönderimden itibaren 10, en son bazı modellerde 20. |
| Neden tohum 42 ve 10 kat | Bu bölme megayak'ın açık OOF kütüphanesi ve heuljax'ın GLR not defteriyle birebir aynı. Böylece dış OOF dosyaları sızıntısız harmanlanabildi. |
| Hedef kodlama | sklearn `TargetEncoder(cv=5, shuffle=True)` her dış katın yalnız eğitim satırlarında; doğrulama ve test o katın kodlayıcısıyla dönüştürülür. |
| `chain` özellikleri | Eğitim satırları iç 5 katla çapraz; doğrulama ve test tüm eğitim satırlarıyla. |
| Erken durdurma | Doğrulama katının AUC'si; en fazla 20.000 ağaç, sabır `int(20 / lr)` tur. |
| Harman ağırlıkları | OOF üzerinde iç içe 5 kat (tohum 42): ağırlıklar 4 katta seçilir, görülmemiş 5. katta ölçülür. |
| Train ve test benzerliği | Düşmanca doğrulama AUC'si 0.50005. |

### Kat sayısının etkisi (CV AUC)

| Model | 5 kat | 10 kat | 20 kat |
|---|---|---|---|
| XGBoost `t1`, `tok`'suz, 3 tohum ortalaması | 0.94619 | 0.94631 | |
| RealMLP, 6 tohum ortalaması | 0.94619 | 0.94624 | |
| CatBoost derinlik 5 | 0.94611 | 0.94622 | |
| XGBoost `t1` tok+chain, iki koşu | | 0.94652 / 0.94648 | 0.94651 / 0.94653 |
| GLR | | 0.94640 | 0.94644 |
| RealMLP + tok, tek tohum | | 0.94637 | 0.94641 |
| Artık XGBoost (GLR logit'inden), iki tohum | | 0.94658 / 0.94654 | 0.94659 / 0.94657 |
| Artık LightGBM | | 0.94656 | 0.94659 |
| LightGBM tok+chain | | 0.94650 | 0.94650 |

5 → 10 kat her modelde yaklaşık +0.0001 getirdi. 10 → 20 kat XGBoost, GLR ve artık modellerde +0.00001 ile
+0.00004 arası getirdi; LightGBM'de değişiklik olmadı.

### CV, public ve private karşılaştırması

CV, harmanın tüm OOF üzerindeki AUC'si. "İç içe" sütunu yukarıdaki iç içe 5 katlı ölçüm; 1–5. gönderimlerde
hesaplanmadı. 1–5. gönderimlerin modelleri 5 katlı.

| No | CV | İç içe | Public | Private |
|---|---|---|---|---|
| 1 | 0.94201 | | 0.94176 | 0.94117 |
| 2 | 0.94575 | | 0.94587 | 0.94495 |
| 3 | 0.94612 | | 0.94625 | 0.94531 |
| 4 | 0.94621 | | 0.94637 | 0.94538 |
| 5 | 0.94626 | | 0.94637 | 0.94537 |
| 6 | 0.94642 | 0.94641 | 0.94642 | 0.94544 |
| 7 | 0.94644 | 0.94643 | 0.94644 | 0.94545 |
| 8 | 0.94645 | 0.94644 | 0.94644 | 0.94545 |
| 9 | 0.94647 | 0.94645 | 0.94645 | 0.94544 |
| 10 | 0.94642 | 0.94642 | 0.94644 | 0.94541 |
| 11 | 0.94661 | 0.94660 | 0.94667 | 0.94561 |
| 12 | 0.94667 | 0.94666 | 0.94671 | 0.94566 |
| 13 | 0.94669 | 0.94667 | 0.94671 | 0.94566 |
| 14 | 0.94669 | 0.94668 | 0.94671 | 0.94566 |
| 15 | 0.94669 | 0.94669 | 0.94671 | 0.94566 |
| 16 | 0.94671 | 0.94670 | 0.94673 | 0.94567 |
| 17 | 0.94672 | 0.94671 | 0.94675 | 0.94569 |
| 18 | 0.94672 | 0.94671 | 0.94676 | 0.94570 |
| 19 | 0.94673 | 0.94672 | 0.94676 | 0.94570 |

Ölçülen uyum:

- **Private sabit bir kaymayla public'in altında.** 2–19. gönderimlerde fark −0.00092 ile −0.00106 arasında
  (1. gönderimde −0.00059). Private, CV'nin de yaklaşık 0.001 altında.
- **Sıralama büyük ölçüde korundu.** 19 gönderim üzerinde Pearson korelasyonu: CV–private 0.9975,
  public–private 0.9997. Spearman: CV–private 0.9854, public–private 0.9845. Yalnızca 6–19. gönderimlerde
  CV–private için Pearson 0.9941, Spearman 0.9675.
- **En iyi private, en iyi CV ve public ile aynı gönderimlerde:** 18 ve 19 (0.94570).
- **Büyük adımlar private'a taşındı:**

  | Adım | CV | Public | Private |
  |---|---|---|---|
  | 10 → 11 (GPT-2 parçaları) | +0.00019 | +0.00023 | +0.00020 |
  | 6 → 19 | +0.00031 | +0.00034 | +0.00026 |
  | 2 → 19 | +0.00098 | +0.00089 | +0.00075 |

- **Küçük CV adımları görünmedi:**

  | Adım | CV | Public | Private |
  |---|---|---|---|
  | 4 → 5 | +0.00005 | 0 | −0.00001 |
  | 7 → 9 | +0.00003 | +0.00001 | −0.00001 |
  | 12 → 15 | +0.00002 (iç içe +0.00003) | 0 | 0 |

## 2. Deney tablosu

### Özellik grupları (5 kat, öğrenme oranı 0.1)

İşe yarayanlar, her satır bir öncekinin üstüne:

| Değişiklik | LightGBM | XGBoost |
|---|---|---|
| `base`: 13 ham sütun | 0.94178 | |
| + `freq`: yaş, gelir ve mesafe değerinin train + test + orijinalde kaç kez geçtiği | 0.94301 | |
| + `dig`: gelir mod 10 / 100 / 1000, mesafenin ondalık hanesi | 0.94381 | |
| + `te1`: her sütunun hedef kodlaması | 0.94543 | 0.94551 |
| + `bins`: gelir // 100, gelir // 1000, tam km; hedef kodlama anahtarı olarak | 0.94570 | 0.94582 |
| + `dig2` (tek tek haneler) + `te3` (yumuşatma auto / 10 / 100) | 0.94573 | 0.94591 |
| XGBoost ayar araması (Optuna, 50 deneme) → `t1` | | 0.94606 |
| + `tok`: gelirin GPT-2 BPE parçaları | 0.94608 | |

`tok`: `" " + str(gelir)` GPT-2 tokenizer'ıyla parçalandı. İlk parça, ilk iki parça ve son parça (parça sayısıyla
birlikte) hedef kodlama anahtarı oldu. Ayrıca bu anahtarların train + test + orijinaldeki log sayımı ve parça sayısı
eklendi. Fikir heuljax'ın GLR not defterinden.

`chain`: gelirde ilk parça → ilk iki parça → tam değer, mesafede tam km → tam değer zincirleme beta hedef oranı
(önsel gücü gelirde 5 / 20 / 80, mesafede 5 / 20). 5 katta katkısı yok (0.94608 → 0.94605), 10 katta var (aşağıda).

Bu düzeyde işe yaramayanlar:

| Değişiklik | Taban → sonuç (LightGBM) | XGBoost |
|---|---|---|
| `orig`: orijinal veri ek eğitim satırı | base 0.94178 → 0.94177; te1 seti 0.94543 → 0.94534 | |
| `ncat`: küçük tam sayılar kategori | 0.94381 → 0.94307 | |
| `te2`: tüm sütun ikililerinin hedef kodlaması | 0.94543 → 0.94522 | |
| `te2s`: düşük kardinaliteli ikililer | 0.94543 → 0.94535 | |
| `ted`: hane izlerinin hedef kodlaması | 0.94543 → 0.94537 | |
| `recipe`: orijinal verinin üretim formülleri | 0.94543 → 0.94544 | 0.94551 → 0.94554 |
| `omean`: orijinal veride değer başına alım oranı | 0.94543 → 0.94543 | 0.94551 → 0.94555 |
| `mb`: `max_bin=1024` | 0.94543 → 0.94544 | 0.94551 → 0.94553 |
| `bins2`: ek ölçekler (gelir / 10, / 500, / 5000; mesafe / 5) | 0.94570 → 0.94570 | 0.94582 → 0.94586 |
| `t1` üstüne `tedig`, `freq2`, `flag` (tek tek ve birlikte) | | 0.94605 → 0.94600–0.94604 |
| `tok` üstüne `mix` | 0.94608 → 0.94597 | |
| `tok` üstüne `tokx` (parça × bağlam hücreleri) | 0.94608 → 0.94591 | |

### Model düzeyi (10 kat, lr 0.02, aksi yazmadıkça)

| Değişiklik | Önce → sonra |
|---|---|
| `tok`, XGBoost `t1`, 3 tohum | 0.94625 / 0.94623 / 0.94625 → 0.94648 / 0.94645 / 0.94647 (ortalama 0.94631 → 0.94652) |
| `tok`, CatBoost d5 | 0.94622 → 0.94645 |
| `tok`, RealMLP tek tohum | 0.9461 → 0.94636 |
| `chain` (`tok` üstüne), XGBoost iki tohum | 0.94648 / 0.94645 → 0.94652 / 0.94648 |
| `chain` (`tok` üstüne), LightGBM | 0.94648 → 0.94650 |
| GLR logit'inden artık model, XGBoost iki tohum | 0.94652 / 0.94648 → 0.94658 / 0.94654 |
| GLR logit'inden artık model, LightGBM | 0.94650 → 0.94656 |
| GLR logit'inden artık model, CatBoost | 0.94648 → 0.94655 |
| Tohum ortalaması, XGBoost `t1` 5 kat | tek tohum 0.94611–0.94613 → 3 tohum 0.94619 |
| Tohum ortalaması, RealMLP 5 kat | tek tohum 0.94594–0.94604 → 6 tohum 0.94619 |
| Tohum ortalaması, artık XGBoost 10 kat | tek tohum 0.94654–0.94658 → 4 tohum 0.94659 (`t2` ayarlarıyla tek koşu 0.94656) |
| LightGBM: açık not defteri ayarları (`pubp`) bizim ayarlar yerine, 5 kat | 0.94593 → 0.94608 |
| CatBoost derinlik, 5 kat lr 0.05 | d6 0.94609, d5 0.94611, d8 (GPU) 0.94592 |
| GLR: kat bölme tohumu 42 / 7 / 11 ortalaması | 0.94640 / 0.94639 / 0.94638 → 0.94643 |
| GLR logit'i üstüne artık MLP (256-64) | 0.94642; 3 bölme tohumu ortalaması 0.94646; 20 kat 0.94647 |
| GLR özellik matrisiyle XGBoost / RealMLP | 0.94636 / 0.94638 |

Artık model: XGBoost `base_margin`, LightGBM `init_score`, CatBoost `baseline` ile GLR logit'inden başlar. Eğitim
satırlarının logit'i, o dış katın eğitim satırlarında iç 5 katla yeniden eğitilen GLR'den gelir; doğrulama ve test
satırlarınınki dış kat GLR'sinden. Fikir goodpjw2008'in "LR-Margin GBDT" not defterinden.

Donanım farkı: XGBoost GPU ve CPU aynı skoru verdi (0.94193 / 0.94192). CatBoost GPU'da aynı katlarda CPU'dan
~0.00026 düşük çıktı; GPU varsayılanı `border_count=128` ile ~0.0008 düşük. Bu yüzden CatBoost CPU'da,
`border_count=254` ile eğitildi.

### Harman düzeyi

| Değişiklik | İç içe CV |
|---|---|
| Yalnız kendi 4 modelimiz (5. gönderim) → açık OOF kütüphaneleri eklenince | 0.94626 → 0.94640 |
| Tepe tırmanma yerine probit(sıra) üstünde lojistik regresyon istifleme | 0.94640 → 0.94639–0.94640; sonra 0.94668–0.94669, aynı düzey |
| Artık modelleri harmana eklemek | +0.00001 |
| 25 açık OOF modelini tek tek eklemek | 3'ü katkı yaptı: residual-stack v19 +0.000014, BlamerX +0.000007, jazivxt zoom-zoom +0.000006; kalanı 0 |
| Dört sınır kuralı (`RULES=1`, bölüm 4) | OOF +0.000001 |
| Ölü bölge kuralı (`DEADZONE=1`, gelir 38.174–41.384) | OOF +0.000004 |

### İşe yaramayanlar (model ve harman düzeyi)

| Deneme | Sonuç |
|---|---|
| Tam veri modeli (katların ortalaması yerine): 4 sahte testli benzetim | tek başına ortalama −0.00004 (üçünde −0.00005 / −0.00007, birinde +0.00003); %25 sıra karışımı +0.00001 |
| Sözde etiketleme, kat başına sızıntısız | ham sütunlarda 0.94164 → 0.94172; XGBoost `t1` 10 kat 0.94625 → 0.94626 |
| Trende %0 / %100 olan gelir ve mesafe aralıklarını kural yapmak | 4 katta bulup 5.'de uygulayınca ≥100 satırlık kurallar −0.00037, ≥300 satır −0.00004 |
| Her ağacı tek sütunla sınırlamak (`solo=1`) | ham sütunlarda +0.00045, bizim sette 0.94576 → 0.94554 |
| RealMLP'ye gelir haneleri ve hedef kodlama eklemek (`PVEP_PLUS=1`) | 10 kat 0.94612, değişmedi; harmanda 8. gönderim public 0.94644, private 0.94545 (7. ile aynı) |
| GLR logit'ini RealMLP'ye özellik olarak vermek | 0.94636 → 0.94596 |
| Kendi MLP'miz (gömmeli) | 12 epoch 0.94525, 30 epoch 0.94520; `lowcat=100` 0.94228; harmanda ağırlık almadı |
| İkinci seviye LightGBM istifleme | 0.94575 |
| Harmandan `init_score` ile başlayıp ham sütunlar + bileşen tahminleriyle düzeltme | 1–8 ağaçta durdu, 0.94644 değişmedi |
| XGBoost'u `tok`'lu setle yeniden ayarlamak (40 deneme) → `t2` | 10 kat tok+chain 0.94650 / 0.94646; `t1` ile aynı düzey |
| GLR L2 = 3 / 10 / 30 | 0.94639 / 0.94640 / 0.94640 |
| LightGBM 20 kat | 0.94650, 10 katla aynı |
| Kaçırılmış desen arayışı (kalibre OOF artıkları) | id sırası AUC 0.49999; birebir tekrar eden satır yok; ardışık id korelasyonu +0.001 (şans sınırı ±0.0012); ~300 gruplamada en yüksek z 2.47 (225 testte şans düzeyi ~2.8) |

## 3. En iyi modellerin hiperparametreleri

Ortak (`train.py`): `n_estimators` / `iterations` 20.000, doğrulama AUC'siyle erken durdurma, sabır `int(20 / lr)`.

**XGBoost `t1`** (lr 0.02; son harmanın en ağır üyesi, 20 kat artık modeli bununla):
`max_depth=4, min_child_weight=1.616, subsample=0.914, colsample_bytree=0.5086, reg_lambda=0.6968,
reg_alpha=0.009999, max_bin=512`; ayrıca `tree_method="hist", enable_categorical=True, max_cat_to_onehot=4`.
Optuna'nın en iyi denemelerinin çoğu derinlik 4, yüksek subsample ve `max_bin` 512–1024 idi; tepe düz
(derinlik 4'lüler 0.94599–0.94606).

**XGBoost `t2`** (lr 0.02): `max_depth=4, min_child_weight=1.066, subsample=0.8726, colsample_bytree=0.5877,
reg_lambda=0.1589, reg_alpha=0.07805, max_bin=1024`.

**LightGBM `pubp`** (lr 0.02; najiama'nın "Pure LGBM" ayarları): `max_depth=5, num_leaves=32,
min_child_samples=10, subsample=0.812763, colsample_bytree=0.30293, reg_alpha=0.07094, reg_lambda=2.03303,
max_bin=1024`; `train.py` varsayılanından `subsample_freq=1`.

**CatBoost `d5`** (lr 0.05, CPU): `depth=5, border_count=254, od_type="Iter", od_wait=400`, kategorik sütunlar
`cat_features` olarak.

**RealMLP** (yekenot'un not defteri, `public/pub_realmlp.py`): `n_ens=8, hidden_dims=[256, 256, 256],
dropout=0.05` (`expm4t` takvimi), SiLU, sayısallar için PBLD gömmesi (`hidden 20, out 5, freq_scale 5.0`,
PReLU, lr çarpanı 0.093), `lr=0.01`, Adam (0.9, 0.99), `flat_anneal` (düz oran 0.3), `weight_decay=0.013`,
EMA 0.997875, gradyan kırpma 1.0, etiket yumuşatma 0.04 (cos), `epochs=2`, `train_bs=256`, erken durdurma yok.

**GLR** (heuljax'ın not defteri, `notebooks/glr/`): L2 = 10, LBFGS 200 iterasyon, 10 dış kat ve 5 iç kat, beta
zinciri önsel güçleri 5 / 20 / 80.

## 4. Final ensemble ve gönderim seçimi

**Yöntem** (`blend_ext.py`):

1. Her OOF ve test tahmini sıraya çevrilir (`rankdata / n`).
2. Tekrar seçilebilir açgözlü tepe tırmanma: en iyi tek modelden başlar, AUC'yi en çok artıran modeli bir kez daha
   ekler, kazanç 1e-7'nin altına inince durur. Ağırlık = seçilme sayısı / toplam.
3. İç içe 5 kat: tepe tırmanma 5 kez OOF satırlarının 4/5'inde çalışır, kalan 1/5'te ölçülür. 18. ve 19.
   gönderimde son ağırlıklar bu 5 ağırlık vektörünün ortalaması (`AVGW=1`).
4. Sınır kuralları (`RULES=1`): sıralamayı grup içinde koruyarak satırları en üste ya da en alta iter. Kurallar
   goodpjw2008'in not defterinden; trende istisnasız:

   | Kural | Train satırı | Alım |
   |---|---|---|
   | gelir ≥ 170.537 → en üst | 393 | 393 |
   | gelir 31.004–41.970 → en alt | 1.257 | 0 |
   | mesafe ≥ 83 km → en alt | 186 | 0 |
   | gelir = 30.000, sübvansiyon yok, (çevre ilgisi 1 ya da menzil kaygısı orta/yüksek) → en alt | 7.157 | 0 |

**19. gönderimin aday listesi:** 28 OOF; komut README'nin "Son gönderimi (19) yeniden üretmek" bölümünde.

**19. gönderimin ağırlıkları:**

`blend_ext.py` aynı komutla yeniden çalıştırıldı; çıktı `pevpsubmission19.csv` ile birebir aynı (en büyük fark 0).
Harman CV 0.94673, iç içe CV 0.94672. 28 adaydan 12'si ağırlık aldı:

| Model | Tek başına OOF AUC | Ağırlık |
|---|---|---|
| Artık XGBoost `t1`, 20 kat (GLR 20 kat logit'inden) | 0.94661 | 0.281 |
| RealMLP + tok, 10 kat | 0.94647 | 0.125 |
| XGBoost, GLR özellik matrisiyle | 0.94636 | 0.094 |
| residual-stack v19 (açık OOF) | 0.94596 | 0.094 |
| heuljax XGB Sample (açık OOF) | 0.94631 | 0.094 |
| Artık CatBoost d5, 10 kat | 0.94655 | 0.074 |
| XGBoost `t1` tok+chain, 10 kat | 0.94656 | 0.059 |
| Artık XGBoost `t1`, 10 kat | 0.94659 | 0.053 |
| XGBoost `t1` tok+chain, 20 kat | 0.94656 | 0.052 |
| LightGBM `pubp` tok+chain, 10 kat | 0.94653 | 0.035 |
| CatBoost d5 tok+chain, 10 kat | 0.94648 | 0.020 |
| RealMLP + tok, 20 kat | 0.94646 | 0.020 |

Ağırlık almayan 16 aday arasında GLR'nin kendisi (0.94643, 20 kat 0.94645), GLR + MLP (0.94646 / 0.94647),
goodpjw2008'in artık LightGBM / XGBoost'u (0.94646) ve artık LightGBM'imiz (0.94657) var. Ağırlık alanların ikisi
tek başına en zayıflar arasında (residual-stack v19 0.94596, heuljax XGB 0.94631).

Kurallar bu harmanda: train'de en üste 393 satır (393 alım), en alta 8.595 satır (0 alım; alt kurallar kesişiyor);
OOF 0.946725 → 0.946727. Testte en üste 156, en alta 3.575 satır gitti.

**Gönderim seçimi:** 19 ve 11 önerildi.

- 19: public'te 18 ile aynı (0.94676), iç içe CV'si biraz daha yüksek (0.94672 / 0.94671).
- 11: yedek olarak. İyi gönderimler içinde en farklısı (sıra korelasyonu 0.9991); 17, 18 ve 19 birbirinin
  neredeyse kopyası olduğu için ikisini birlikte seçmek yedek işlevi görmezdi.

Kaggle'da hangi ikisinin işaretlendiği API çıktısında yok, bu dosyada doğrulanmadı. Sonuç: 19 private 0.94570,
11 private 0.94561. En yüksek private 0.94570 idi (18 ve 19).

## 5. Dış ve orijinal veri kullanımı

**Orijinal veri** (`itzzomkar/ev-adoption-behavior-and-range-anxiety`, 10.000 satır, `original.csv`):

| Kullanım | Nerede | Ölçülen etki |
|---|---|---|
| Ek eğitim satırı (`orig`) | `train.py` | yok ya da negatif: 0.94178 → 0.94177; 0.94543 → 0.94534 |
| Değer başına alım oranı (`omean`) | `train.py` | LightGBM 0, XGBoost +0.00004 |
| Frekans sayımları (`freq`) ve `tok` sayımları | sayım tabanına train + test + orijinal birlikte girer | orijinalsiz sürümü ayrıca ölçülmedi |
| GPT-2 parça eşlemesi | `gpt2tok.py` train + test + orijinal değerlerinin hepsini eşler | eşleme tablosu, model değil |
| `recipe`: orijinal verinin üretim formülleri (C. Deotte'nin EDA'sından) | `train.py` | LightGBM +0.00001, XGBoost +0.00003 |

GLR not defteri orijinal verinin yolunu tanımlıyor ama okumuyor; yalnız train + test kullanıyor.

**Dış tahmin dosyaları** (aynı 10 katlı bölmeyle üretilmiş açık OOF'lar, `ext/` altında):

| Kaynak | Ne |
|---|---|
| `megayak/s6e9-six-feature-views-oof-library` | altı özellik hattı ve RealMLP OOF'ları (her biri 0.94608–0.94628) |
| `najiama/s6e9-oof` | Pure LGBM sürümleri, üçlü TE XGBoost (0.94533–0.94624) |
| `legtarrr/s6e9-residual-stack-oof` | residual-stack v19 ve jazivxt zoom-zoom |
| `medvax/s6e9-medvax-blamerx-oof-predictions` | BlamerX OOF'ları |
| goodpjw2008 "LR-Margin GBDT + OOF Stack" çıktıları | artık LightGBM / XGBoost (0.94646), GLM |
| heuljax XGB Sample | 0.94631; harmandaki en farklı üye (sıra korelasyonu 0.9918) |
| BlamerX pencere kodlamalı XGBoost | `blamerx_win` |

Açık OOF'ların test tahminlerinde bileşenler arası korelasyon, OOF'takinden en fazla 0.0012 farklıydı.

**Önceden eğitilmiş GPT-2 tokenizer'ı** (Hugging Face `gpt2`): yalnız gelir değerlerini parçalamak için,
internetli bir Kaggle CPU not defterinde.

## 6. Yeniden kullanılabilir kod

| Dosya | Ne işe yarar |
|---|---|
| `train.py` | Özellik grupları + LightGBM / XGBoost / CatBoost / MLP; K katlı CV, `oof_<etiket>.npy` ve `pred_<etiket>.npy` yazar. Ayarlar komut satırından (`max_depth=4,...`), ayrıca `seed=`, `folds=`, `name=`, `trials=` (Optuna), `margin=1` (artık model), `full=` / `hold=` (tam veri benzetimi), `pl=` (sözde etiket), `solo=1`. |
| `kaggle_run.py` | `train.py` işlerini özel bir Kaggle not defterinde (T4 GPU ya da `PVEP_CPU=1` ile CPU) çalıştırır, günlüğü basar, `.npy` çıktılarını indirir. `PVEP_SLUG` ile aynı anda birden çok not defteri. |
| `blend.py` | Kendi `oof_/pred_*.npy` dosyalarımız üzerinde basit tepe tırmanma; virgülle verilen etiketleri (tohumlar) önce ortalar. |
| `blend_ext.py` | Kendi modellerimiz + `ext/` altındaki açık OOF'lar; sıra uzayında tepe tırmanma, iç içe CV, `AVGW`, `STACK`, `RULES`, `DEADZONE`, `FULLMIX`. Dış dosyaları id ile hizalar ve hedefle karşılaştırarak doğrular. |
| `gpt2tok.py` | Sayısal değerleri GPT-2 BPE parçalarına eşler (`gpt2_income_tokens.csv`, `gpt2_commute_tokens.csv`). |
| `public/pub_realmlp.py` | yekenot'un RealMLP'si; `PVEP_SEED`, `PVEP_FOLDS`, `PVEP_TOK`, `PVEP_PLUS`, `PVEP_FULL`, `PVEP_HOLD`, `PVEP_MARGIN`. |
| `public/pub_lgbm.py`, `public/pub_xgb.py` | najiama ve evgendvorkin not defterlerinin uyarlamaları (CV 0.94609 / 0.94607). |
| `notebooks/` | GLR ailesi: GLR, sızıntısız GLR logit'leri (`glr_margins.npz`, 10 ve 20 kat), GLR + MLP, GLR özellikli XGBoost ve RealMLP; her klasörde `kernel-metadata.json`. |
| `requirements.txt` | Sürümler: pandas 3.0.6, numpy 2.4.6, scikit-learn 1.9.1, lightgbm 4.7.0, xgboost 3.2.0, catboost 1.2.10, optuna 5.0.0. |

Başka yarışmaya taşırken koda gömülü, bu yarışmaya özel yerler:

- `train.py`: `TARGET`, `CATS`, `NUMS`, hedefin `"Yes"` ile ikiliye çevrilmesi; `dig`, `bins`, `flag`, `recipe`,
  `tok`, `chain`, `mix` grupları gelir ve mesafe sütun adlarını doğrudan kullanır.
- `blend_ext.py`: hedef ve id sütunu adları, `ext/` altındaki dosya yolları, `RULES` ve `DEADZONE` eşikleri.
- `kaggle_run.py`: yarışma slug'ı (`playground-series-s6e9`), orijinal veri seti, Kaggle kullanıcı adı ve
  `kernel_sources` slug'ları.
- `gpt2tok.py` ve GLR not defterleri: sütun adları ve Kaggle girdi yolları.
