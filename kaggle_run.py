"""train.py'yi Kaggle'da özel bir GPU not defterinde (T4) çalıştırır, bitince oof_/pred_*.npy
dosyalarını buraya indirir. xgb ve cat GPU'da, lgbm CPU'da eğitilir (GPU etiketine _gpu eklenir).
Kullanım: python kaggle_run.py "xgb base,freq,dig,te1 0.02" "cat base,freq,dig,te1 0.05"
          python kaggle_run.py "xgb base,freq,dig,te1 0.1 trials=40"   # Optuna araması
          python kaggle_run.py --fetch   # son sürümü bekle, günlüğü göster, çıktıyı indir
PVEP_SLUG=<ad> başka bir not defteri (ve kernel_build_<ad> klasörü) kullanır, PVEP_CPU=1 GPU'suz."""
import json
import os
import shutil
import sys
import time

from kaggle import api

SLUG = os.environ.get("PVEP_SLUG", "pvep-train")  # aynı anda ikinci iş için başka not defteri
BUILD = "kernel_build" if SLUG == "pvep-train" else f"kernel_build_{SLUG}"
CPU = os.environ.get("PVEP_CPU") == "1"  # GPU yuvası harcamayan CPU not defteri (lgbm için)
REF = f"{api.get_config_value('username')}/{SLUG}"
GROUPS = {"base", "freq", "freq2", "dig", "dig2", "recipe", "omean", "flag", "te1", "te3", "bins",
          "bins2", "tedig", "te2", "te2s", "ted", "ncat", "orig", "mb", "tok", "tokx", "mix", "chain"}
TOKSRC = "hasancmert/pvep-gpt2tok"  # gpt2_income_tokens.csv'yi üreten not defteri (girdi olarak bağlanır)

# Not defterinde çalışan kod; başına JOBS ve TRAIN (train.py'nin metni) eklenir.
BODY = """import glob, os, shutil, subprocess, sys
import pandas, sklearn, lightgbm, xgboost, catboost
print(*(f"{m.__name__} {m.__version__}" for m in (pandas, sklearn, lightgbm, xgboost, catboost)),
      flush=True)

def find(name):
    return sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))[0]

os.makedirs("/tmp/pvep", exist_ok=True)
os.chdir("/tmp/pvep")  # çıktıya yalnızca .npy dosyaları gitsin
for src, dst in [("train.csv", "train.csv"), ("test.csv", "test.csv"),
                 ("EV_Adoption_and_Range_Anxiety_Dataset.csv", "original.csv")]:
    os.symlink(find(src), dst)
for f in glob.glob("/kaggle/input/**/gpt2_income_tokens.csv", recursive=True)[:1]:
    os.symlink(f, "gpt2_income_tokens.csv")
with open("train.py", "w") as f:
    f.write(TRAIN)
env = dict(os.environ, PVEP_GPU="1" if shutil.which("nvidia-smi") else "0")
for job in JOBS:
    print(">>", job, flush=True)
    subprocess.run([sys.executable, "train.py", *job.split()], env=env, check=True)
    for f in glob.glob("*.npy"):
        shutil.move(f, "/kaggle/working/")
"""


def push(jobs):
    for job in jobs:
        model, groups, *rest = job.split()  # rest: [öğrenme oranı [ayarlar]]
        if model not in ("lgbm", "xgb", "cat", "nn") or not set(groups.split(",")) <= GROUPS \
                or len(rest) > 2 or rest and float(rest[0]) <= 0:
            sys.exit(f"hatalı iş: {job!r}")
    shutil.rmtree(BUILD, ignore_errors=True)
    os.makedirs(BUILD)
    with open("train.py", encoding="utf-8") as f:
        train = f.read()
    with open(f"{BUILD}/run.py", "w", encoding="utf-8") as f:
        f.write(f"JOBS = {jobs!r}\nTRAIN = {train!r}\n" + BODY)
    meta = dict(id=REF, title=SLUG, code_file="run.py", language="python", kernel_type="script",
                is_private=True, enable_gpu=not CPU, enable_tpu=False, enable_internet=False,
                **({} if CPU else {"machine_shape": "NvidiaTeslaT4"}),
                competition_sources=["playground-series-s6e9"],
                dataset_sources=["itzzomkar/ev-adoption-behavior-and-range-anxiety"],
                kernel_sources=[TOKSRC] if any(g in job.split()[1].split(",") for job in jobs
                                               for g in ("tok", "tokx", "mix", "chain")) else [],
                model_sources=[])
    with open(f"{BUILD}/kernel-metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    r = api.kernels_push(BUILD)
    while r is not None and r.error and "Maximum batch GPU" in r.error:  # GPU slotları dolu: bekle
        time.sleep(60)
        r = api.kernels_push(BUILD)
    if r is None or r.error:
        sys.exit(f"push hatası: {r and r.error}")
    print(f"sürüm {r.versionNumber}: {r.url}", flush=True)


def fetch():
    last = None
    while True:
        s = api.kernels_status(REF)
        if s.status.name != last:
            last = s.status.name
            print(time.strftime("%H:%M:%S"), last, flush=True)
        if last in ("COMPLETE", "ERROR", "CANCEL_ACKNOWLEDGED"):
            break
        time.sleep(30)
    log = api.kernels_logs(REF)
    try:
        entries = json.loads(log)
        out = "".join(e["data"] for e in entries if e.get("stream_name") == "stdout")
        err = "".join(e["data"] for e in entries if e.get("stream_name") == "stderr")
    except (ValueError, TypeError, KeyError):
        out, err = log, ""
    print(out)
    if last != "COMPLETE":
        sys.exit(f"{err[-3000:]}\n{last}: {s.failure_message}")
    try:
        files, _ = api.kernels_output(REF, f"{BUILD}/out", file_pattern=r"\.npy$", force=True)
    except OSError as e:  # dosyalar www.kaggleusercontent.com'dan iner
        sys.exit(f"çıktı indirilemedi ({type(e).__name__}); dosyalar Kaggle'da duruyor. "
                 "www.kaggleusercontent.com erişimi açılınca: python kaggle_run.py --fetch")
    for f in files:
        if f.endswith(".npy"):  # günlük (pvep-train.log) kernel_build/out'ta kalır
            shutil.move(f, os.path.basename(f))
            print("indirildi:", os.path.basename(f))


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        sys.exit(__doc__)
    if args != ["--fetch"]:
        push(args)
        time.sleep(30)  # yeni sürüm başlamadan öncekinin durumu okunmasın
    fetch()
