"""Дворкин Евгений (evgendvorkin) - "S6E9 Single XGB CV: 0.94607" açık Kaggle not defterinin uyarlaması:
https://www.kaggle.com/code/evgendvorkin/s6e9-single-xgb-cv-0-94607
Değişen yalnızca: yazarın görsel stil kütüphanesi yerine boş fonksiyonlar, veri yolları (Kaggle'da glob,
yerelde depo kökü) ve çıktı adları (oof_/pred_pub_xgb.npy). 10 katlı CV (train.py'nin 5 katından farklı
katlar, OOF yine dürüst) ve GPU (device='cuda') kullanır."""
import glob, os


def find(name):
    hits = sorted(glob.glob(f"/kaggle/input/**/{name}", recursive=True))
    return hits[0] if hits else ("original.csv" if name.startswith("EV_") else name)


def md_metric(k, v):
    print(f"{k}: {v}", flush=True)


h1 = h2 = info_card = pretty_df = lambda *a, **k: None
import sys

import gc
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from scipy import stats

warnings.filterwarnings('ignore')

T0 = time.perf_counter()
def log(msg):
    print(f'[{time.perf_counter() - T0:7.1f}s] {msg}', flush=True)

h1("🚗 EV Adoption Prediction — S6E9")
info_card("✅ Setup Complete", "Импорты выполнены, стиль загружен.", style="vi")

# ----- cell
train = pd.read_csv(find('train.csv'))
test = pd.read_csv(find('test.csv'))
orig = pd.read_csv(find('EV_Adoption_and_Range_Anxiety_Dataset.csv'))

h2("📥 Data Loading")
info_card("Dataset Shapes", 
          f"Train: {train.shape} | Test: {test.shape} | Orig: {orig.shape}", 
          style="vi")
pretty_df(train.head(3))

# ----- cell
cat_cols = ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
            'Subsidy_Available', 'Range_Anxiety_Level']

for col in cat_cols:
    mapping = {val: i for i, val in enumerate(train[col].unique())}
    train[f'LE_{col}'] = train[col].map(mapping)
    test[f'LE_{col}'] = test[col].map(mapping)

# Удаляем исходные категориальные колонки
train = train.drop(columns=cat_cols)
test = test.drop(columns=cat_cols)

# Убираем id
train = train.drop(columns=['id'])
test = test.drop(columns=['id'])

# Разделяем X и y
X = train.drop(columns=['Will_Buy_EV'])
y = train['Will_Buy_EV']

h2("🔧 Preprocessing")
info_card("✅ Base Preprocessing Done", f"X shape: {X.shape} | Test shape: {test.shape}", style="vi")

# ----- cell
# Interaction признаки на основе EDA
X['Env_Concern_x_Subsidy'] = X['Environmental_Concern_Level'] * X['LE_Subsidy_Available']
test['Env_Concern_x_Subsidy'] = test['Environmental_Concern_Level'] * test['LE_Subsidy_Available']

X['Env_Concern_x_Income'] = X['Environmental_Concern_Level'] * X['Annual_Income_USD']
test['Env_Concern_x_Income'] = test['Environmental_Concern_Level'] * test['Annual_Income_USD']

X['Subsidy_x_Income'] = X['LE_Subsidy_Available'] * X['Annual_Income_USD']
test['Subsidy_x_Income'] = test['LE_Subsidy_Available'] * test['Annual_Income_USD']

X['Env_Concern_x_Commute'] = X['Environmental_Concern_Level'] * X['Daily_Commute_km']
test['Env_Concern_x_Commute'] = test['Environmental_Concern_Level'] * test['Daily_Commute_km']

X['Home_Charging_x_Subsidy'] = X['LE_Home_Charging_Possible'] * X['LE_Subsidy_Available']
test['Home_Charging_x_Subsidy'] = test['LE_Home_Charging_Possible'] * test['LE_Subsidy_Available']

X['Range_Anxiety_x_Subsidy'] = X['LE_Range_Anxiety_Level'] * X['LE_Subsidy_Available']
test['Range_Anxiety_x_Subsidy'] = test['LE_Range_Anxiety_Level'] * test['LE_Subsidy_Available']

X['Income_per_Concern'] = X['Annual_Income_USD'] / (X['Environmental_Concern_Level'] + 1)
test['Income_per_Concern'] = test['Annual_Income_USD'] / (test['Environmental_Concern_Level'] + 1)

X['Commute_per_Concern'] = X['Daily_Commute_km'] / (X['Environmental_Concern_Level'] + 1)
test['Commute_per_Concern'] = test['Daily_Commute_km'] / (test['Environmental_Concern_Level'] + 1)

print(f"✅ Добавлено interaction признаков")
print(f"X shape: {X.shape}")

# ----- cell
# Числовые признаки для извлечения цифр
digit_num_cols = [
    'Age', 'Annual_Income_USD', 'Daily_Commute_km',
    'Number_of_Cars_Owned', 'Charging_Stations_Near_Home',
    'Charging_Stations_Near_Work', 'Environmental_Concern_Level'
]

# Извлекаем цифры в позициях от -4 до 3
for col in digit_num_cols:
    for k in range(-4, 4):
        new_col = f'{col}_digit{k}'
        X[new_col] = (X[col].fillna(0) // (10**k) % 10).astype('int8')
        test[new_col] = (test[col].fillna(0) // (10**k) % 10).astype('int8')

print(f"✅ Добавлено digit features: {len(digit_num_cols) * 8}")
print(f"X shape: {X.shape} | Test shape: {test.shape}")

# ----- cell
# Считаем частоты по train+test для каждой колонки (включая digit и LE)
for column in X.columns:
    freq_map = pd.concat([X[column], test[column]], axis=0).value_counts(normalize=True).to_dict()
    X[f'{column}_freq'] = X[column].map(freq_map).astype('float32').values
    test[f'{column}_freq'] = test[column].map(freq_map).astype('float32').values

print(f"✅ Добавлено frequency features: {len([c for c in X.columns if c.endswith('_freq')])}")
print(f"X shape: {X.shape} | Test shape: {test.shape}")

# ----- cell
object_cols = X.select_dtypes(include=['object']).columns.tolist()
X = X.drop(columns=object_cols)
test = test.drop(columns=object_cols)

print(f"Удалено object: {len(object_cols)}")
print(f"X: {X.shape} | test: {test.shape}")

# ----- cell
# Удаляем константные
const_cols = [c for c in X.columns if X[c].nunique() == 1]
X = X.drop(columns=const_cols)
test = test.drop(columns=const_cols)

# Удаляем коррелирующие (corr=1)
corr_matrix = X.corr().abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_tri.columns if any(upper_tri[col] == 1.0)]
X = X.drop(columns=to_drop)
test = test.drop(columns=to_drop)

print(f"✅ Удалено: {len(const_cols)} константных + {len(to_drop)} коррелирующих")
print(f"X: {X.shape} | test: {test.shape}")

del corr_matrix, upper_tri
gc.collect()

# ----- cell
import xgboost as xgb
from sklearn.preprocessing import TargetEncoder

# Конвертируем таргет в 0/1
y = y.map({'No': 0, 'Yes': 1})

# Параметры
params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'learning_rate': 0.005,
    'max_depth': 7,
    'min_child_weight': 10,
    'subsample': 0.9,
    'colsample_bytree': 0.9,
    'device': 'cuda',
    'reg_alpha': 0.071,
    'reg_lambda': 2.0,
    'max_bin': 1024,
    'n_estimators': 10000,
    'early_stopping_rounds': 500,
    'nthread': -1,
    'deterministic_histogram': True,
}

# Колонки для Triple TE — 17 (13 core + 4 smooth keys)
te_cols = ['Age', 'Annual_Income_USD', 'Daily_Commute_km', 'Number_of_Cars_Owned',
           'Charging_Stations_Near_Home', 'Charging_Stations_Near_Work',
           'Environmental_Concern_Level', 'LE_Gender', 'LE_City_Type', 'LE_Current_Car_Type',
           'LE_Home_Charging_Possible', 'LE_Subsidy_Available', 'LE_Range_Anxiety_Level',
           'income_exact_int', 'income100_floor', 'income1000_floor', 'commute_integer']

# Читаем исходные данные для TE (один раз)
train_te_raw = pd.read_csv(find('train.csv'))
test_te_raw = pd.read_csv(find('test.csv'))
for col in ['Gender', 'City_Type', 'Current_Car_Type', 'Home_Charging_Possible',
            'Subsidy_Available', 'Range_Anxiety_Level']:
    mapping = {val: i for i, val in enumerate(train_te_raw[col].unique())}
    train_te_raw[f'LE_{col}'] = train_te_raw[col].map(mapping)
    test_te_raw[f'LE_{col}'] = test_te_raw[col].map(mapping)

# === Smooth Keys (Markus's method) ===
train_te_raw['income_exact_int'] = np.floor(train_te_raw['Annual_Income_USD']).astype(int).astype(str)
test_te_raw['income_exact_int'] = np.floor(test_te_raw['Annual_Income_USD']).astype(int).astype(str)

train_te_raw['income100_floor'] = np.floor(train_te_raw['Annual_Income_USD'] / 100).astype(int).astype(str)
test_te_raw['income100_floor'] = np.floor(test_te_raw['Annual_Income_USD'] / 100).astype(int).astype(str)

train_te_raw['income1000_floor'] = np.floor(train_te_raw['Annual_Income_USD'] / 1000).astype(int).astype(str)
test_te_raw['income1000_floor'] = np.floor(test_te_raw['Annual_Income_USD'] / 1000).astype(int).astype(str)

train_te_raw['commute_integer'] = np.floor(train_te_raw['Daily_Commute_km']).astype(int).astype(str)
test_te_raw['commute_integer'] = np.floor(test_te_raw['Daily_Commute_km']).astype(int).astype(str)

# StratifiedKFold — 10 фолдов
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Массивы для сохранения
oof_preds = np.zeros(len(X))
test_preds = np.zeros(len(test))
feature_importance = None

log("Starting 10-fold CV with Triple TE (17 cols)...")

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train = X.iloc[train_idx].copy()
    X_val = X.iloc[val_idx].copy()
    X_test_fold = test.copy()
    y_train = y.iloc[train_idx]
    y_val = y.iloc[val_idx]
    
    # Triple Target Encoding (auto, 10, 100) по 17 колонкам
    for smooth_val, smooth_name in [('auto', 'auto'), (10.0, '10'), (100.0, '100')]:
        te = TargetEncoder(shuffle=True, cv=5, smooth=smooth_val, random_state=42)
        X_train_enc = te.fit_transform(train_te_raw.iloc[train_idx][te_cols], y_train).astype('float32')
        X_val_enc = te.transform(train_te_raw.iloc[val_idx][te_cols]).astype('float32')
        X_test_enc = te.transform(test_te_raw[te_cols]).astype('float32')
        
        for i, col in enumerate(te_cols):
            X_train[f'TE_{col}_{smooth_name}'] = X_train_enc[:, i]
            X_val[f'TE_{col}_{smooth_name}'] = X_val_enc[:, i]
            X_test_fold[f'TE_{col}_{smooth_name}'] = X_test_enc[:, i]
    
    current_seed = 42
    params['random_state'] = current_seed
    params['seed'] = current_seed
    
    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=100
    )
    
    oof_preds[val_idx] = model.predict_proba(X_val)[:, 1]
    test_preds += model.predict_proba(X_test_fold)[:, 1] / skf.n_splits
    
    if feature_importance is None:
        feature_importance = np.zeros(X_train.shape[1])
    feature_importance += model.feature_importances_ / skf.n_splits
    
    fold_auc = roc_auc_score(y_val, oof_preds[val_idx])
    h2(f"Fold {fold}/10")
    md_metric("Fold AUC", f"{fold_auc:.5f}")
    md_metric("Seed", str(current_seed))
    md_metric("Elapsed", f"{time.perf_counter() - T0:.1f}s")

# Overall OOF score
oof_auc = roc_auc_score(y, oof_preds)
h2("🏆 Final OOF Performance")
md_metric("OOF AUC", f"{oof_auc:.5f}")

# Feature importance
importance_df = pd.DataFrame({
    'feature': X_train.columns,
    'importance': feature_importance
}).sort_values('importance', ascending=False)

h2("📊 Top 15 Features")
pretty_df(importance_df.head(15))

# ----- cell
np.save('oof_pub_xgb.npy', oof_preds)
np.save('pred_pub_xgb.npy', test_preds)

h2("💾 Save Predictions")
info_card("✅ Predictions Saved", 
          f"OOF: {oof_preds.shape} | Test: {test_preds.shape} | OOF AUC: {oof_auc:.5f}", 
          style="vi")

# ----- cell
submission = pd.DataFrame({
    'id': pd.read_csv(find('test.csv'))['id'],
    'Will_Buy_EV': test_preds
})

submission.to_csv('submission.csv', index=False)

h2("📤 Submission Created")
info_card("✅ Submission Saved", f"Shape: {submission.shape}", style="vi")
pretty_df(submission.head(10))

# ----- cell


# ----- cell
