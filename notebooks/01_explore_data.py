# RSNA Knee Abnormality Detection: data exploration.
#
# Run in a Kaggle notebook that has the competition data attached.
# CPU is enough and internet can stay off. Everything printed is also written
# to /kaggle/working/explore_summary.txt so it can be downloaded and shared.

import os
import sys
import time
import glob
import json
import random
import platform
import subprocess
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

SUMMARY_PATH = os.environ.get("SUMMARY_PATH", "/kaggle/working/explore_summary.txt")
WALK_TIME_BUDGET_S = 900      # stop the full file walk after this long
DICOM_STUDIES_TO_SAMPLE = 150  # studies whose series headers are aggregated
DICOM_TIME_BUDGET_S = 600
SEED = 0
random.seed(SEED)


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)

    def flush(self):
        for st in self.streams:
            st.flush()


os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
_summary_file = open(SUMMARY_PATH, "w")
_orig_stdout = sys.stdout
sys.stdout = Tee(_orig_stdout, _summary_file)

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)
pd.set_option("display.max_colwidth", 80)


def section(title):
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def human(n):
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024
    return f"{n:.1f}PB"


# ----------------------------------------------------------------------------
section("1. Environment")
print("python", platform.python_version())
for mod in ["torch", "torchvision", "timm", "pydicom", "SimpleITK", "nibabel",
            "cv2", "sklearn", "transformers", "lightgbm", "xgboost",
            "pylibjpeg", "gdcm", "monai"]:
    try:
        m = __import__(mod)
        print(f"{mod:14s} {getattr(m, '__version__', 'installed')}")
    except Exception as e:
        print(f"{mod:14s} MISSING ({type(e).__name__})")
try:
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total",
                          "--format=csv,noheader"], capture_output=True,
                         text=True, timeout=20).stdout.strip() or "no GPU")
except Exception:
    print("no GPU")
print("CPU count:", os.cpu_count())
try:
    print(subprocess.run(["df", "-h", "/kaggle/working", "/kaggle/input"],
                         capture_output=True, text=True).stdout)
    print(subprocess.run(["free", "-g"], capture_output=True, text=True).stdout)
except Exception:
    pass

# ----------------------------------------------------------------------------
section("2. Input mounts")
INPUT_ROOT = os.environ.get("INPUT_ROOT", "/kaggle/input")
for p in sorted(glob.glob(INPUT_ROOT + "/*")) + sorted(glob.glob(INPUT_ROOT + "/*/*")):
    print(p)

comp_dirs = [p for p in glob.glob(INPUT_ROOT + "/*") + glob.glob(INPUT_ROOT + "/*/*")
             if "knee" in os.path.basename(p.rstrip("/")).lower() and os.path.isdir(p)]
DATA_ROOT = comp_dirs[0] if comp_dirs else INPUT_ROOT
print("\nDATA_ROOT =", DATA_ROOT)

# ----------------------------------------------------------------------------
section("3. Directory tree (depth 4, first entries per level)")


def show_tree(root, depth=0, max_depth=4, max_children=3):
    try:
        entries = sorted(os.scandir(root), key=lambda e: e.name)
    except Exception as e:
        print("  " * depth + f"[error {e}]")
        return
    dirs = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]
    indent = "  " * depth
    if files:
        shown = files[:max_children]
        for f in shown:
            try:
                sz = human(f.stat().st_size)
            except Exception:
                sz = "?"
            print(f"{indent}{f.name}  ({sz})")
        if len(files) > len(shown):
            print(f"{indent}... {len(files) - len(shown)} more files "
                  f"(total {len(files)})")
    for d in dirs[:max_children]:
        print(f"{indent}{d.name}/")
        if depth + 1 < max_depth:
            show_tree(d.path, depth + 1, max_depth, max_children)
    if len(dirs) > max_children:
        print(f"{indent}... {len(dirs) - max_children} more dirs (total {len(dirs)})")


show_tree(DATA_ROOT)

# ----------------------------------------------------------------------------
section("4. Full walk: file counts, extensions, sizes, depth pattern")
t0 = time.time()
ext_count = Counter()
ext_size = Counter()
depth_count = Counter()
top_level_count = Counter()
n_files = 0
total_size = 0
complete = True
all_small_files = []  # non-image files, likely tables/reports
dicom_like_dirs = defaultdict(int)  # leaf dir -> number of files
for dirpath, dirnames, filenames in os.walk(DATA_ROOT):
    if time.time() - t0 > WALK_TIME_BUDGET_S:
        complete = False
        break
    dirnames.sort()
    rel = os.path.relpath(dirpath, DATA_ROOT)
    depth = 0 if rel == "." else rel.count(os.sep) + 1
    top = rel.split(os.sep)[0]
    for fn in filenames:
        ext = os.path.splitext(fn)[1].lower() or "<none>"
        try:
            sz = os.path.getsize(os.path.join(dirpath, fn))
        except Exception:
            sz = 0
        ext_count[ext] += 1
        ext_size[ext] += sz
        depth_count[depth] += 1
        top_level_count[top] += 1
        n_files += 1
        total_size += sz
        if ext in (".dcm", "<none>", ".dicom"):
            dicom_like_dirs[dirpath] += 1
        elif ext not in (".png", ".jpg", ".jpeg", ".npy", ".npz", ".nii", ".gz"):
            all_small_files.append((os.path.join(dirpath, fn), sz))
print(f"walk complete: {complete}  ({time.time() - t0:.0f}s)")
print(f"files: {n_files:,}   total size: {human(total_size)}")
print("\nby extension:")
for ext, c in ext_count.most_common(20):
    print(f"  {ext:10s} {c:>10,}  {human(ext_size[ext])}")
print("\nfiles by depth below DATA_ROOT:", dict(sorted(depth_count.items())))
print("\nfiles by top-level entry:")
for k, c in top_level_count.most_common(20):
    print(f"  {k:40s} {c:,}")
print("\nnon-image files (tables, reports, etc.):")
for p, sz in sorted(all_small_files)[:60]:
    print(f"  {os.path.relpath(p, DATA_ROOT)}  ({human(sz)})")
if len(all_small_files) > 60:
    print(f"  ... {len(all_small_files) - 60} more")
if dicom_like_dirs:
    counts = np.array(list(dicom_like_dirs.values()))
    print(f"\nleaf dirs holding DICOM-like files: {len(counts):,}")
    print("files per such dir: min %d  median %d  mean %.1f  max %d" % (
        counts.min(), np.median(counts), counts.mean(), counts.max()))
    ex = list(dicom_like_dirs.keys())[:5]
    print("examples:")
    for d in ex:
        print("  ", os.path.relpath(d, DATA_ROOT), dicom_like_dirs[d])

# ----------------------------------------------------------------------------
section("5. Tables (csv / parquet / json / jsonl / tsv)")
tables = {}
table_paths = [p for p, _ in all_small_files
               if p.lower().endswith((".csv", ".parquet", ".tsv", ".json", ".jsonl"))]
for p in table_paths:
    name = os.path.relpath(p, DATA_ROOT)
    try:
        if p.endswith(".parquet"):
            df = pd.read_parquet(p)
        elif p.endswith(".tsv"):
            df = pd.read_csv(p, sep="\t")
        elif p.endswith(".jsonl"):
            df = pd.read_json(p, lines=True)
        elif p.endswith(".json"):
            with open(p) as f:
                obj = json.load(f)
            if isinstance(obj, list):
                df = pd.json_normalize(obj)
            else:
                print(f"\n--- {name}: JSON object with keys {list(obj)[:30]}")
                print(json.dumps(obj, ensure_ascii=False)[:1500])
                continue
        else:
            df = pd.read_csv(p, low_memory=False)
    except Exception as e:
        print(f"\n--- {name}: could not read ({e})")
        continue
    tables[name] = df
    print(f"\n--- {name}   shape={df.shape}")
    info = pd.DataFrame({
        "dtype": df.dtypes.astype(str),
        "n_null": df.isna().sum(),
        "n_unique": df.nunique(dropna=True),
        "example": [str(df[c].dropna().iloc[0])[:60] if df[c].notna().any() else ""
                    for c in df.columns],
    })
    print(info.to_string())
    print("\nhead(3):")
    print(df.head(3).astype(str).apply(lambda col: col.str.slice(0, 60)).to_string())
    # small-cardinality columns: value counts (labels, languages, series types)
    for c in df.columns:
        nu = df[c].nunique(dropna=False)
        if 1 < nu <= 15:
            vc = df[c].value_counts(dropna=False)
            print(f"\n  value_counts[{c}]: " +
                  ", ".join(f"{k}={v}" for k, v in vc.items()))

# ----------------------------------------------------------------------------
section("6. Label columns (binary / small-integer columns)")
for name, df in tables.items():
    lab_cols = []
    for c in df.columns:
        s = df[c].dropna()
        if len(s) and pd.api.types.is_numeric_dtype(s) and s.nunique() <= 4 \
                and set(s.unique()).issubset({-1, 0, 1, 2, 3}):
            lab_cols.append(c)
    if not lab_cols:
        continue
    print(f"\n--- {name}: {len(lab_cols)} label-like columns, {len(df)} rows")
    rows = []
    for c in lab_cols:
        s = df[c]
        rows.append({"col": c, "non_null": int(s.notna().sum()),
                     **{f"={v}": int((s == v).sum()) for v in sorted(s.dropna().unique())}})
    print(pd.DataFrame(rows).fillna(0).to_string(index=False))

# ----------------------------------------------------------------------------
section("7. Free-text columns and report samples")
for name, df in tables.items():
    for c in df.columns:
        if not (df[c].dtype == object or pd.api.types.is_string_dtype(df[c])):
            continue
        s = df[c].dropna().astype(str)
        if not len(s):
            continue
        mean_len = s.str.len().mean()
        if mean_len < 80:
            continue
        print(f"\n--- {name} :: column '{c}'  non-null={len(s)}  "
              f"mean_len={mean_len:.0f}  max_len={s.str.len().max()}")
        idx = random.sample(range(len(s)), min(4, len(s)))
        for i in idx:
            txt = s.iloc[i].replace("\r", " ")
            print(f"\n  [sample row {s.index[i]}]\n  " + txt[:700].replace("\n", "\n  ")
                  + (" ..." if len(txt) > 700 else ""))

# report files outside tables (txt / md / xml)
txt_files = [p for p, _ in all_small_files if p.lower().endswith((".txt", ".md", ".xml"))]
if txt_files:
    print(f"\n{len(txt_files)} text files found. Samples:")
    for p in random.sample(txt_files, min(3, len(txt_files))):
        with open(p, errors="replace") as f:
            txt = f.read()
        print(f"\n  [{os.path.relpath(p, DATA_ROOT)}]\n  " + txt[:700].replace("\n", "\n  "))

# ----------------------------------------------------------------------------
section("8. DICOM structure")
try:
    import pydicom
except Exception as e:
    pydicom = None
    print("pydicom missing:", e)

HEADER_TAGS = ["Modality", "Manufacturer", "ManufacturerModelName",
               "MagneticFieldStrength", "SeriesDescription", "ProtocolName",
               "SequenceName", "ScanningSequence", "MRAcquisitionType",
               "Rows", "Columns", "PixelSpacing", "SliceThickness",
               "SpacingBetweenSlices", "ImageOrientationPatient",
               "PhotometricInterpretation", "BitsStored", "RepetitionTime",
               "EchoTime", "BodyPartExamined", "Laterality", "PatientSex",
               "PatientAge", "StudyInstanceUID", "SeriesInstanceUID",
               "SeriesNumber", "InstanceNumber"]


def orientation_name(iop):
    try:
        iop = np.array([float(x) for x in iop])
        normal = np.abs(np.cross(iop[:3], iop[3:]))
        return ["sagittal", "coronal", "axial"][int(np.argmax(normal))]
    except Exception:
        return "?"


if pydicom and dicom_like_dirs:
    series_dirs = sorted(dicom_like_dirs)
    # Group leaf dirs by their parent (usually the study) to sample whole studies.
    by_parent = defaultdict(list)
    for d in series_dirs:
        by_parent[os.path.dirname(d)].append(d)
    parents = sorted(by_parent)
    print(f"parent dirs of DICOM leaf dirs: {len(parents):,}  "
          f"(leaf dirs per parent: {Counter(len(v) for v in by_parent.values()).most_common(8)})")

    # Detailed look at 3 studies
    for parent in random.sample(parents, min(3, len(parents))):
        print(f"\n### {os.path.relpath(parent, DATA_ROOT)}")
        for d in by_parent[parent]:
            files = sorted(os.listdir(d))
            fp = os.path.join(d, files[0])
            try:
                ds = pydicom.dcmread(fp, stop_before_pixels=True, force=True)
            except Exception as e:
                print(f"  {os.path.basename(d)}: {len(files)} files, read error {e}")
                continue
            vals = {t: str(getattr(ds, t, ""))[:60] for t in HEADER_TAGS}
            print(f"  series dir {os.path.basename(d)}: {len(files)} files, "
                  f"first file {files[0]}, orientation="
                  f"{orientation_name(getattr(ds, 'ImageOrientationPatient', None))}")
            print("    " + "; ".join(f"{k}={v}" for k, v in vals.items() if v))
            print("    transfer syntax:", getattr(ds.file_meta, "TransferSyntaxUID", "?"))

    # Aggregate over a sample of studies
    t0 = time.time()
    agg = Counter()
    orient = Counter()
    n_slices = defaultdict(list)
    sizes = Counter()
    manu = Counter()
    field = Counter()
    tsyntax = Counter()
    series_per_study = []
    sampled = random.sample(parents, min(DICOM_STUDIES_TO_SAMPLE, len(parents)))
    n_done = 0
    for parent in sampled:
        if time.time() - t0 > DICOM_TIME_BUDGET_S:
            break
        series_per_study.append(len(by_parent[parent]))
        for d in by_parent[parent]:
            files = os.listdir(d)
            try:
                ds = pydicom.dcmread(os.path.join(d, sorted(files)[0]),
                                     stop_before_pixels=True, force=True)
            except Exception:
                agg["<unreadable>"] += 1
                continue
            desc = str(getattr(ds, "SeriesDescription", "<none>")).strip()
            agg[desc] += 1
            o = orientation_name(getattr(ds, "ImageOrientationPatient", None))
            orient[o] += 1
            n_slices[o].append(len(files))
            sizes[(getattr(ds, "Rows", None), getattr(ds, "Columns", None))] += 1
            manu[str(getattr(ds, "Manufacturer", "?"))] += 1
            field[str(getattr(ds, "MagneticFieldStrength", "?"))] += 1
            tsyntax[str(getattr(ds.file_meta, "TransferSyntaxUID", "?"))] += 1
        n_done += 1
    print(f"\naggregated over {n_done} studies ({time.time() - t0:.0f}s)")
    print("series per study:", Counter(series_per_study).most_common())
    print("orientation:", orient.most_common())
    for o, v in n_slices.items():
        v = np.array(v)
        print(f"  slices per {o} series: min {v.min()} median {np.median(v):.0f} max {v.max()}")
    print("\nSeriesDescription (top 40):")
    for k, c in agg.most_common(40):
        print(f"  {c:5d}  {k}")
    print("\nimage sizes:", sizes.most_common(10))
    print("manufacturer:", manu.most_common(10))
    print("field strength:", field.most_common(10))
    print("transfer syntax:", tsyntax.most_common(10))

    # Pixel decoding check (codec support matters offline)
    print("\npixel decode check:")
    for parent in sampled[:5]:
        d = by_parent[parent][0]
        fp = os.path.join(d, sorted(os.listdir(d))[0])
        try:
            arr = pydicom.dcmread(fp, force=True).pixel_array
            print(f"  ok  {arr.shape} {arr.dtype} min={arr.min()} max={arr.max()}  "
                  f"{os.path.relpath(fp, DATA_ROOT)}")
        except Exception as e:
            print(f"  FAIL {type(e).__name__}: {str(e)[:150]}")
else:
    print("No DICOM-like files found (or pydicom missing).")

# ----------------------------------------------------------------------------
section("9. Links between tables and image folders")
leaf_names = {os.path.basename(d) for d in dicom_like_dirs}
parent_names = {os.path.basename(os.path.dirname(d)) for d in dicom_like_dirs}
for name, df in tables.items():
    for c in df.columns:
        vals = set(df[c].dropna().astype(str).head(5000))
        if not vals or len(vals) < 5:
            continue
        hit_study = len(vals & parent_names) / len(vals)
        hit_series = len(vals & leaf_names) / len(vals)
        if hit_study > 0.2 or hit_series > 0.2:
            print(f"{name} :: {c}  matches study-dir names {hit_study:.0%}, "
                  f"series-dir names {hit_series:.0%}")

section("Done")
print("Summary written to", SUMMARY_PATH)
sys.stdout = _orig_stdout
_summary_file.close()
