# HANDOFF: Kaggle Playground S6E9 → NFL Big Data Bowl 2027

Kaynaklar: bu repo (`train.py`, `blend_ext.py`, `notebooks/`, README.md, LESSONS.md), writeup taslağı ve bu sohbet.
Kaydı olmayan her sayı "kayıt yok" diye yazıldı. Satır numaraları `66132c7` commit'ine göre; ayrıntılı tablolar LESSONS.md'de.

## 1. Problem, veri, metrik ve son skorlar

Sentetik tablo verisinde ikili sınıflandırma: `Will_Buy_EV` (Yes = %17.46), metrik ROC AUC. Train 668.665, test 286.571 satır,
13 özellik (7 sayısal, 6 kategorik) + 10.000 satırlık orijinal veri (543 NaN); train/test'te NaN ve tekrar eden satır yok.

| | OOF CV | İç içe CV | Public LB | Private LB |
|---|---|---|---|---|
| Son gönderim (19): 28 adaydan 12'si ağırlıklı harman | 0.94673 | 0.94672 | 0.94676 | 0.94570 |
| En iyi tek model: artık XGBoost `t1`, 20 kat (2 tohum ortalaması) | 0.94661 | – | kayıt yok | kayıt yok |

Public sıra 133 / 3.576 (1 Ekim 2026'da indirilen leaderboard). Private sıra: kayıt yok. Kaggle'da işaretlenen son iki
gönderim: kayıt yok (öneri 19 + 11; private skorları 0.94570 / 0.94561).

## 2. Pipeline

- **Temizlik:** ayrı adım yok. Train + test'in 955.236 satırında NaN ve birebir tekrar eden satır yok; imputation yapılmadı.
  Kategorikler `pd.Categorical`, hedef `Yes` → 1. Orijinal veri yalnız etiketsiz sayımlara girdi (NaN'leri ayrı kod alır).
- **Özellikler** (final set `base,freq,dig,dig2,te1,te3,bins,tok,chain`; `train.py:76–193`):
  - `freq`: yaş, gelir ve mesafe değerinin train + test + orijinalde kaç kez geçtiği (etiketsiz).
  - `dig`, `dig2`: gelir mod 10 / 100 / 1000, mesafenin ondalığı, tek tek haneler (sentetik üreteç izleri).
  - `te1`, `te3`: 13 sütunun hedef kodlaması (`TargetEncoder`, yumuşatma auto / 10 / 100), kat içinde.
  - `bins`: gelir // 100, gelir // 1000 ve tam km, hedef kodlama anahtarı olarak.
  - `tok`: gelirin GPT-2 BPE parçaları (ilk, ilk iki, son): anahtar + etiketsiz log sayım + parça sayısı.
  - `chain`: kaba → ince anahtar zincirinde beta küçültmeli hedef oranı (gelir: ilk parça → ilk iki → tam; önsel 5 / 20 / 80).
- **Modeller:** XGBoost `t1` (derinlik 4, lr 0.02), LightGBM `pubp`, CatBoost d5 (CPU), RealMLP (PyTorch), GLR (L2 = 10 ridge
  lojistik regresyon, heuljax), GLR logit'inden başlayan artık XGBoost / LightGBM / CatBoost. Finalde ağırlık alan dış OOF'lar:
  residual-stack v19 ve heuljax XGB (0.094'er).
- **CV:** `StratifiedKFold(K, shuffle=True, random_state=42)`, K 5 → 10 → 20; tüm modeller aynı katlarda. Erken durdurma
  doğrulama katının AUC'siyle (en çok 20.000 ağaç, sabır `int(20 / lr)`).
- **Harman:** OOF'lar sıraya çevrilir → tekrar seçilebilir açgözlü tepe tırmanma → ağırlıklar iç içe 5 katta seçilip ortalanır
  (`AVGW=1`) → dört sınır kuralı (`RULES=1`). En büyük ağırlık artık XGBoost 20 kat (0.281).

## 3. CV'yi en çok iyileştiren adımlar

Katkılar farklı düzeylerde ölçüldü (tek model 5 kat lr 0.1, tek model 10 kat lr 0.02 ya da harman iç içe CV); toplanamaz.

| # | Adım | Katkı (AUC) | Ölçüm |
|---|---|---|---|
| 1 | `te1` hedef kodlama | +0.00162 | LightGBM 5 kat (0.94381 → 0.94543) |
| 2 | `freq` sayımları | +0.00123 | LightGBM 5 kat (0.94178 → 0.94301) |
| 3 | `dig` hane izleri | +0.00080 | LightGBM 5 kat (0.94301 → 0.94381) |
| 4 | `tok` GPT-2 parçaları | +0.00035; 10 katta ≈ +0.00021 … +0.00026 | LightGBM 5 kat; XGBoost / CatBoost / RealMLP 10 kat |
| 5 | `bins` kaba anahtarlar | +0.00027 / +0.00031 | LightGBM / XGBoost 5 kat |
| 6 | Tohum ortalaması | RealMLP +0.00015 … +0.00025; XGBoost +0.00006 … +0.00008 | 5 kat, 6 / 3 tohum |
| 7 | Optuna ile XGBoost ayarı (`t1`) | +0.00015 | 5 kat lr 0.1, 50 deneme |
| 8 | LightGBM'de açık not defteri ayarları (`pubp`) | +0.00015 | 5 kat lr 0.02 |
| 9 | Harmana açık OOF kütüphaneleri | +0.00014 | iç içe CV (0.94626 → 0.94640) |
| 10 | 5 → 10 kat | XGBoost +0.00012, CatBoost +0.00011, RealMLP +0.00005 | XGBoost 3, RealMLP 6 tohum ortalaması |
| 11 | `dig2` + `te3` | +0.00003 / +0.00009 | LightGBM / XGBoost 5 kat |
| 12 | GLR logit'inden artık GBDT (`margin=1`) | +0.00006 … +0.00007 / model; harmana +0.00001 | 10 kat; iç içe CV |
| 13 | `chain` (`tok` üstüne); 10 → 20 kat | +0.00002 … +0.00004; +0.00001 … +0.00004 | 10 kat; 20 kat (LightGBM'de 0) |

GLR'nin harmana tek başına katkısı ölçülmedi: kayıt yok (12. gönderimde başka modellerle birlikte girdi, ağırlığı 0.286).
Doğrusal GLR (10 kat 0.94640) en iyi düz GBDT'nin (XGBoost, 10 kat 0.94652) 0.00012 altında; GLR'den başlayan artık XGBoost 0.94658.

**İşe yaramayanlar:**
- Orijinal veriyi ek eğitim satırı yapmak (0.94178 → 0.94177; 0.94543 → 0.94534); küçük tam sayıları kategori yapmak (−0.00074).
- Sütun ikilisi hedef kodlaması (`te2` −0.00021, `te2s` −0.00008), hane TE (`ted` −0.00006), `tokx` −0.00017, `mix` −0.00011.
- Etkisi ≈ 0: `recipe`, `omean`, `max_bin=1024`, `bins2`, `tedig` / `freq2` / `flag` (−0.00005 … +0.00004).
- Tam veri modeli (kat ortalaması yerine): 4 sahte testte ortalama −0.00004. Sözde etiketleme: +0.00001.
- Örneklemde bulunan %0 / %100 aralık kuralları: görülmemiş katta −0.00037 (≥ 100 satırlık kurallar).
- Tek sütunlu ağaçlar (`solo=1`): ham sütunlarda +0.00045, zengin özellik setinde −0.00022.
- GLR logit'ini sinir ağına özellik vermek: 0.94636 → 0.94596. Kendi gömmeli MLP'miz 0.94525, harmanda ağırlık 0.
- İkinci seviye LightGBM istifleme 0.94575; LR istifleme tepe tırmanmayla aynı; harman üstüne düzeltici model 1–8 ağaçta durdu.
- `tok` sonrası yeniden ayar (`t2`), GLR L2 3 / 10 / 30, LightGBM 20 kat: değişiklik yok. 25 dış OOF'tan 22'si: katkı 0.

## 4. Sızıntı kontrolleri ve CV–LB uyumu

- **Kat içi üretim:** hedef-türevli her özellik dış katın yalnız eğitim satırlarıyla üretildi. `TargetEncoder(cv=5)` eğitim
  satırlarını kendi içinde çapraz kodlar (`train.py::fold_data`); `chain` iç 5 kat; GLR etiket özellikleri iç 5 kat (`build_fold`).
- **Artık modeller:** doğrulama / test logit'i dış kat GLR'sinden, eğitim satırlarınınki iç 5 kat GLR'sinden (`glr_margin/glrm.ipynb`).
- **Tek bölme:** megayak OOF kütüphanesinin `fold` sütunu bizim `StratifiedKFold(10, shuffle, 42)` bölmemizle birebir aynı
  (çapraz tabloyla doğrulandı); GLR not defteri aynı çağrıyı kullanıyor. Dış OOF'lar id ile hizalanıyor; megayak'ta hedef assert'li.
- **Harman seçimi:** ağırlıklar iç içe 5 katta seçildi; tüm-OOF CV ile iç içe CV farkı 6–19. gönderimlerde 0 … 0.00002.
- **Etiketsiz ama transdüktif:** `freq` ve `tok` sayımları test ve orijinal satırları da sayar (etiket kullanmaz).
- **Ölçülmeyen:** erken durdurma doğrulama katının kendisiyle yapıldı; bunun OOF'a etkisi: kayıt yok.
- **Kayma / desen taraması:** düşmanca doğrulama AUC 0.50005; id sırası–hedef AUC 0.49999; ardışık id korelasyonu +0.001
  (şans sınırı ±0.0012); ~300 gruplamada en yüksek z 2.47 (225 testte şans düzeyi ~2.8).
- **CV → private:** private, 2–19. gönderimlerde public'in sabit 0.00092 … 0.00106, CV'nin 0.00080 … 0.00104 altında.
  Sıra korundu: CV–private Pearson 0.9975, Spearman 0.9854 (19 gönderim). En iyi private (0.94570) 18 ve 19'da: ikisi en
  iyi public'li, 19 en iyi CV'li gönderim.
- **Adım büyüklüğü:** +0.00019'luk adım (10 → 11) private'a +0.00020 olarak taşındı. +0.00005 ve altındaki adımlar tutarsız:
  4 → 5, 7 → 9 ve 12 → 15 private'ta görünmedi; 16 → 19 (+0.00002) +0.00003 olarak yansıdı.

## 5. Tekrar kullanılabilir kod

| Yol | Ne işe yarar |
|---|---|
| `train.py:197`, `train.py::fold_data` (:214) | Tek sabit bölme; kat içinde TE (`TargetEncoder(cv=5)`) ve `chain` çapraz üretimi |
| `train.py::chain_feats` (:201) | Kaba → ince anahtar zincirinde beta küçültmeli hedef oranı (logit); herhangi bir hiyerarşiye uygulanır |
| `train.py::lgbm` / `xgbm` / `cat` (:252 / :271 / :292) | Erken durdurmalı GBDT sarmalayıcıları; `mg=` ile başlangıç logit'inden artık model |
| `train.py::solo` (:243) | Her ağacı tek sütunla sınırlar (toplamsal model) |
| `train.py:396` (`trials=N`) | Optuna TPE ile XGBoost ayar araması (her deneme 5 kat CV) |
| `train.py:455` (`hold=h`, `full=f`) | Tam veri modeli ile kat ortalamasını sahte testte karşılaştırır |
| `train.py:76–193` (grup bayrakları) | Özellik gruplarını adla aç/kapat; `oof_/pred_<etiket>.npy` → ablation tablosu |
| `blend_ext.py::hill` (:188), :220–228 | Tekrar seçilebilir tepe tırmanma; iç içe 5 kat ve ağırlık ortalaması (`AVGW`) |
| `blend_ext.py:208` (`STACK=1`) | probit(sıra) üstünde lojistik istifleme, iç içe ölçümle |
| `blend_ext.py::aligned` (:91) | Dış tahmin dosyalarını id ile hizalar |
| `blend.py` | Kendi OOF'larımız için sade tepe tırmanma; virgüllü etiketleri (tohumlar) önce ortalar |
| `kaggle_run.py::push` / `fetch` (:53 / :86) | Betiği Kaggle not defterinde (GPU / CPU) çalıştırır, günlüğü basar, çıktıyı indirir |
| `notebooks/glr/glr.ipynb`: `build_fold`, `fit_glm` | Kat içi çapraz özellik matrisi; torch LBFGS ile L2 lojistik regresyon |
| `notebooks/glr_margin/glrm.ipynb` (`# pvep` satırları) | Sızıntısız logit üretimi (iç 5 kat) → `glr_margins.npz` |
| `public/pub_realmlp.py::RealMLP_TD_Classifier` (:596) | RealMLP (PyTorch); tohum ve kat ortam değişkenleriyle |
| `gpt2tok.py` | Sayıları GPT-2 BPE parçalarına eşler |

Tüm model çıktıları (npy, parquet, gönderimler) özel Kaggle veri setinde: `hasancmert/pvep-model-outputs`.

## 6. NFL Big Data Bowl 2027'ye taşınabilirlik

NFL koşulları: LB yok; jüri puanlı ≤ 2000 kelimelik yazı + public notebook; 10 Hz Combine verisi (x, y, s, a, dir) → NFL
maç performansı; ~510 oyuncu, 5 pozisyon grubu; yorumlanabilirlik skordan önemli.

| Yöntem | Etiket | Gerekçe |
|---|---|---|
| Tek sabit bölme, tüm modeller aynı katlarda | [TAŞINIR] | Karşılaştırma için şart; oyuncu başına birden çok satır varsa oyuncu bazında gruplanmalı, pozisyona göre katmanlanmalı. |
| Daha çok kat (5 → 10 → 20) | [TAŞINIR] | Burada 5 → 10 kat +0.00005 … +0.00012 verdi; 10 katta ~51 oyuncu/kat kalır, tekrarlı K-kat ile ortalama ± sapma raporlanmalı. |
| İç içe CV ile seçim (ağırlık, ayar) | [TAŞINIR] | Küçük örneklemde seçim iyimserliği büyür; seçilen her şey görülmemiş dış katta ölçülmeli. |
| Hedef-türevli özellikleri kat içinde çapraz üretmek | [TAŞINIR] | Sızıntı kuralı veri boyutundan bağımsız; pozisyon/oyuncu düzeyi hedef ortalamaları da kat içinde hesaplanmalı. |
| Yüksek kardinaliteli hedef kodlama (`te1`, `te3`, `bins`) | [TAŞINMAZ] | Burada binlerce değerli sütunlar vardı (gelirde 13.214); 5 pozisyon grubu için one-hot ya da pozisyon içi ölçekleme yeter. |
| Sentetik üreteç izleri (`freq`, `dig`, `dig2`, `tok`, sınır kuralları) | [TAŞINMAZ] | Sentetik üretecin izlerine dayanıyor (hane izleri, GPT-2 tabanlı üreteç varsayımı); gerçek sensör verisinde karşılığı yok. |
| Hiyerarşik beta küçültme (`chain_feats`) | [TAŞINIR] | Pozisyon grubu → oyuncu gibi iç içe yapıda az gözlemli grupları üst gruba çeken, açıklanabilir ampirik Bayes. |
| L2 düzenlileştirilmiş lojistik / doğrusal model (GLR) | [TAŞINIR] | Katsayılar okunur, küçük n'de düzenlileştirme gerekli; burada en iyi GBDT'nin yalnız 0.00012 altında kaldı. |
| Doğrusal logit'ten başlayan artık GBDT (`margin=1`) | [TAŞINIR] | Doğrusal modelin açıklamadığı payı ayrıca ölçer (burada 0.94640 → 0.94658); iç kat logit kuralıyla sızıntısız. |
| Çok ağaçlı GBDT + Optuna araması | [TAŞINMAZ] | Burada bile ayar +0.00015 getirdi ve tepe düzdü; 510 oyuncuda ayar seçimi aşırı uyum riski taşır, yorumu zorlaştırır. |
| Tek sütunlu ağaçlar (`solo=1`, toplamsal model) | [TAŞINIR] | Her özelliğin etkisi tek başına çizilebilir (GAM benzeri); burada ham sütunlarda +0.00045 verdi. |
| RealMLP / sinir ağları | [TAŞINMAZ] | ~510 örnek için parametre çok, yorumu zor; burada da en iyi tek model değildi (0.94647 vs 0.94661). |
| Tohum ortalaması | [TAŞINIR] | Ucuz varyans azaltma; NFL'de asıl kullanımı tekrarlı CV / bootstrap ile katsayı kararlılığını göstermek. |
| Sıra uzayında harman, LR istifleme | [TAŞINMAZ] | LB yok; 12 modelli harman en iyi tek modelden yalnız +0.00012 (0.94661 → 0.94673) ve jüriye açıklanamaz. |
| Dış OOF / başkalarının tahminleri | [TAŞINMAZ] | Jüri özgün ve açıklanabilir analizi puanlıyor; dış tahmin harmanı yöntemi açıklanamaz kılar. |
| Örneklemden kural madenciliği | [TAŞINMAZ] | Görülmemiş katta −0.00037 kaybettirdi; 510 oyuncuda daha kırılgan. |
| Sözde etiketleme | [TAŞINMAZ] | Etiketsiz test kümesi / LB yok; burada da yalnız +0.00001. |
| Tam veri benzetimi (`hold=h`, `full=f`) | [TAŞINIR] | "Son modeli tüm veriyle eğitelim mi" sorusunu ölçerek cevaplar. |
| Düşmanca doğrulama ve artık grup taraması | [TAŞINIR] | Pozisyon grupları arası kaymayı sınar; çok sayıda alt grup testinde şans düzeyini (burada 225 testte z ≈ 2.8) yazıda belirtmeli. |
| Erken durdurmayı doğrulama katıyla yapmak | [TAŞINMAZ] | Etkisi burada ölçülmedi; küçük n'de aynı katı hem durdurma hem ölçüm için kullanmak iyimserliği büyütür. |
| Ablation düzeni (grup bayrakları, etiketli OOF, README tablosu) | [TAŞINIR] | Yazıdaki "hangi sensör özelliği ne kattı" tablosunu doğrudan üretir. |
| `kaggle_run.py` ile Kaggle'da koşturma | [TAŞINIR] | Public notebook teslimi için; `is_private`, slug ve veri kaynakları değiştirilmeli. |
| Writeup iskeleti (TL;DR, doğrulama, işe yaramayanlar, teşekkür) | [TAŞINIR] | Jüri yazısının bölümleri hazır; 2000 kelime sınırına göre kısaltılmalı. |

10 Hz zaman serisinden özellik çıkarma bu projede yoktu (tablo verisi); bu iş için taşınacak kod yok.
