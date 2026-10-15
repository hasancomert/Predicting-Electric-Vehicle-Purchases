"""Gelir ve mesafe değerlerinin GPT-2 BPE parçaları -> gpt2_income_tokens.csv, gpt2_commute_tokens.csv.
Veri üreticisi GPT-2 tabanlı olabilir (sayıları parça parça yazar); fikir P. B. Elefante'nin "KPS6E09
Generator-Aware Ridge Logistic Regression" not defterinden. transformers ve GPT-2 tokenizer'ı ister; Kaggle'da
internetli CPU not defteri olarak çalıştırılır (hasancmert/pvep-gpt2tok; kernel-metadata.json'da
enable_internet true, enable_gpu false), çıktısı kaggle_run.py'de girdi olarak bağlanır."""
import glob

import pandas as pd
from transformers import AutoTokenizer


def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    return hits[0] if hits else ("original.csv" if name.startswith("EV_") else name)


tr, te = pd.read_csv(find("train.csv")), pd.read_csv(find("test.csv"))
orig = pd.read_csv(find("EV_Adoption_and_Range_Anxiety_Dataset.csv"))
tok = AutoTokenizer.from_pretrained("gpt2")
for col, fmt, out in [("Annual_Income_USD", lambda v: str(int(v)), "gpt2_income_tokens.csv"),
                      ("Daily_Commute_km", lambda v: repr(float(v)), "gpt2_commute_tokens.csv")]:
    vals = pd.concat([tr[col], te[col], orig[col]]).dropna().unique()
    rows = []
    for v in vals:
        ids = tok(" " + fmt(v)).input_ids
        rows.append((v, len(ids), ids[0], ids[1] if len(ids) > 1 else -1, ids[2] if len(ids) > 2 else -1,
                     ids[-1], "|".join(tok.convert_ids_to_tokens(ids))))
    df = pd.DataFrame(rows, columns=["value", "ntok", "t1", "t2", "t3", "last", "pieces"])
    df.to_csv(out, index=False)
    print(col, len(df), "değer; parça sayısı dağılımı:", df.ntok.value_counts().to_dict())
    print(df.sample(8, random_state=0).to_string())
