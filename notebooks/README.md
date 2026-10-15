# Kaggle not defterleri (GLR ailesi)

Son gönderimlerde (12-19) kullanılan, `train.py` dışında çalışan not defterleri. Hepsi
[heuljax (P. B. Elefante) — KPS6E09 Generator-Aware Ridge Logistic Regression](https://www.kaggle.com/code/heuljax/kps6e09-generator-aware-ridge-logistic-regression)
not defterine dayanır; `glr/` onun değiştirilmemiş hâlidir, diğerleri `# pvep` yorumlu küçük eklerle türetilmiştir.
Her klasörde `kernel-metadata.json` var: `kaggle kernels push -p notebooks/<klasör>` ile kendi hesabınızda çalışır
(`id`'deki kullanıcı adını değiştirin). GPT-2 tokenizer'ı için internet açık olmalı.

| Klasör | Ne yapar | Çıktı | CV |
|---|---|---|---|
| `gpt2tok/` | gelir/mesafe değerlerinin GPT-2 BPE parça eşlemesi (kod: kökteki `gpt2tok.py`) | `gpt2_income_tokens.csv` | - |
| `glr/` | üreticiye duyarlı ridge lojistik regresyon, 10 kat | OOF/test parquet | 0.94640 |
| `glr_margin/` | aynı GLR + sızıntısız logit'ler (eğitim satırları iç 5 kat) | `glr_margins.npz` | 0.94640 |
| `glr_margin_f20/` | 20 katlı sürümü | `glr_margins_f20.npz` | 0.94644 |
| `glr_mlp/` | GLR logit'i üstüne artık MLP (2 tohum) | OOF/test parquet | 0.94642 |
| `glr_xgb/` | aynı özellik matrisiyle XGBoost (GPU) | OOF/test parquet | 0.94636 |
| `glr_realmlp/` | aynı özellik matrisiyle RealMLP (GPU; sınıflar `public/pub_realmlp.py`'den) | OOF/test parquet | 0.94638 |

Varyantlar notebook metninde tek satır değiştirilerek üretildi (`blend_ext.py` bunları `ext/pbe-*` altında arar):
- kat bölme tohumu: `N_FOLDS, INNER_FOLDS, RANDOM_STATE = 10, 5, 42` → `..., 7` / `..., 11` (`ext/pbe-glr-s7`, `-s11`, `ext/pbe-mlp-s7` ...)
- 20 kat: `= 20, 5, 42` (`ext/pbe-glr-f20`, `ext/pbe-mlp-f20`)
- L2: `L2 = 10.0` → `3.0` / `30.0` (fark yok)

`glr_margin*` çıktıları `train.py ... margin=1` (artık XGBoost/LightGBM/CatBoost) ve RealMLP `PVEP_MARGIN=1` için
`kaggle_run.py` tarafından girdi olarak bağlanır (`hasancmert/pvep-glr-margin`, `-f20`).
