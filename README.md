# RSNA Knee Abnormality Detection

Kaggle yarışması: https://www.kaggle.com/competitions/rsna-knee-abnormality-detection

Kod bu depoda yazılır, Kaggle Notebooks üzerinde çalıştırılır.

## Notebook'lar

| Dosya | Amaç | Donanım |
|---|---|---|
| `notebooks/01_explore_data.ipynb` | Veri yapısını çıkarır (dosyalar, tablolar, raporlar, DICOM serileri) | CPU yeterli |

## Kaggle'da çalıştırma

1. Yarışma sayfasında **Code > New Notebook**. Yarışma verisi otomatik bağlanır.
2. **File > Import Notebook** ile `notebooks/01_explore_data.ipynb` dosyasını yükle
   (ya da `notebooks/01_explore_data.py` içeriğini tek bir hücreye yapıştır).
3. **Run All**.
4. Bitince sağdaki **Output** panelinden `explore_summary.txt` dosyasını indir.

Kaggle API anahtarını (`kaggle.json`) asla depoya ekleme.
