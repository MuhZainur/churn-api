import os, json, time, warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
from pycaret.classification import (
    setup, compare_models, tune_model, finalize_model,
    save_model, pull, blend_models, predict_model, get_config
)
import mlflow
from sklearn.metrics import roc_auc_score

# -------------------------------
# 0) Konfigurasi & data loading
# -------------------------------
DATA_PATH = os.getenv("CHURN_DATA_PATH", "training/data/churn.csv")
TARGET = os.getenv("TARGET_COL", "churn")
RANDOM_STATE = int(os.getenv("SEED", "42"))
EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT", "bank-churn-pycaret")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")  # bisa diganti ke server MLflow
mlflow.set_tracking_uri(TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH)
else:
    # fallback: cari dataset umum di folder project (silakan ganti sesuai lokasi kamu)
    raise FileNotFoundError(
        f"Data tidak ditemukan di {DATA_PATH}. Set env CHURN_DATA_PATH atau taruh CSV di training/data/churn.csv"
    )

# drop kolom ID jika ada
for id_col in ["customer_id", "CustomerId", "id"]:
    if id_col in df.columns:
        df = df.drop(columns=[id_col])

# pastikan target biner 0/1
if df[TARGET].dtype != "int64" and df[TARGET].dtype != "int32":
    # kalau bernilai Yes/No atau string lain
    df[TARGET] = df[TARGET].astype("category").cat.codes

# -------------------------------
# 1) Setup PyCaret (schema stabil)
# -------------------------------
# deteksi tipe
num_cols = df.select_dtypes(include=['int64','float64','int32','float32']).columns.tolist()
cat_cols = df.select_dtypes(include=['object','category','bool']).columns.tolist()
if TARGET in num_cols: num_cols.remove(TARGET)
if TARGET in cat_cols: cat_cols.remove(TARGET)

# Fix imbalance (SMOTE), normalisasi numerik, one-hot untuk kategorikal,
# multicollinearity handling, dan stratified CV.
s = setup(
    data=df,
    target=TARGET,
    train_size=0.8,
    fold=5,
    fold_strategy="stratifiedkfold",
    session_id=RANDOM_STATE,
    imputation_type="simple",
    numeric_imputation="median",
    categorical_imputation="most_frequent",
    normalize=True,
    fix_imbalance=True,                   # SMOTE by default
    remove_multicollinearity=True,
    multicollinearity_threshold=0.95,
    feature_selection=False,
    categorical_features=cat_cols or None,
    numeric_features=num_cols or None,
    verbose=False,
    use_gpu=False,
    n_jobs=1,
    verbose=False,
)

# -------------------------------
# 2) Model selection & tuning
# -------------------------------
# --- pilih 1 model terbaik, tanpa blending, prioritas metrik ROC_AUC agar konsisten dengan API
best = compare_models(sort="AUC", verbose=False)
tuned = tune_model(best, optimize="AUC", choose_better=True, verbose=False)
final_model = finalize_model(tuned)

# -------------------------------
# 3) Evaluasi hold-out (test set)
# -------------------------------
holdout_df = predict_model(final_model)  # otomatis pada holdout dari setup
auc_holdout = roc_auc_score(holdout_df[TARGET], holdout_df["prediction_score"])

# Tarik metrik cross-validated dari pull()
cv_results = pull().copy()   # setelah tune_model / finalize, pull mengandung metrik
metrics = {
    "AUC_holdout": float(auc_holdout),
    "AUC_CV_mean": float(cv_results.get("AUC").mean() if "AUC" in cv_results else np.nan),
    "Accuracy_CV_mean": float(cv_results.get("Accuracy").mean() if "Accuracy" in cv_results else np.nan),
}

# -------------------------------
# 4) MLflow logging
# -------------------------------
with mlflow.start_run(run_name=f"pycaret_{int(time.time())}") as run:
    mlflow.log_params({
        "train_size": 0.8,
        "folds": 5,
        "fix_imbalance": True,
        "normalize": True,
        "remove_multicollinearity": True,
        "random_state": RANDOM_STATE
    })
    mlflow.log_metrics(metrics)

    # log feature schema untuk stabilitas API
    feature_cols = [c for c in df.columns if c != TARGET]
    mlflow.log_dict({"feature_columns": feature_cols}, "feature_schema.json")

    # Simpan pipeline (PyCaret sudah mengemas preprocessor+model)
    save_path = "training/pycaret_pipeline"
    save_model(final_model, save_path)   # menghasilkan training/pycaret_pipeline.pkl
    # Log sebagai artifact MLflow
    mlflow.log_artifact(f"{save_path}.pkl", artifact_path="model")

# -------------------------------
# 5) Export ke bundle kompatibel FastAPI
# -------------------------------
# Ambil pipeline ter-finalize dari config
pipe = get_config("prep_pipe")  # preprocessor
model = get_config("trained_model")
# NOTE: `save_model` di atas sudah menyatukan pipeline; tapi untuk konsisten dengan backend kita,
# kita cukup memuat file .pkl yang dibuat PyCaret sebagai 1 pipeline sklearn.
# Agar isi bundle konsisten, kita muat lagi dan bungkus ke dict kita.

import joblib, sklearn
full_pipeline = joblib.load(f"{save_path}.pkl")  # sklearn-like pipeline

bundle = {
    "model": full_pipeline,                     # seluruh pipeline (preprocess + estimator)
    "target_encoder": None,                     # tidak dipakai krn target sudah 0/1
    "feature_columns": feature_cols,            # simpan urutan/daftar fitur
    "sklearn_version": sklearn.__version__,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "training_metrics": metrics
}

# simpan ke path backend agar memicu deploy backend
os.makedirs("backend", exist_ok=True)
joblib.dump(bundle, "backend/model_bundle.joblib")
print("Saved bundle -> backend/model_bundle.joblib")
