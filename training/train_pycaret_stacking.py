# INI ADALAH STACKING MODEL JIKA INGIN MENGUJI DENGAN INI CUKUP BUANG TULISAN STAKING PADA NAMA FILE
import os, json, time, warnings
warnings.filterwarnings("ignore")
#
import pandas as pd
import numpy as np
from pycaret.classification import (
    setup, compare_models, tune_model, finalize_model,
    save_model, pull, blend_models, predict_model, get_config,stack_models
)
import mlflow
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression

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
    fix_imbalance=True,
    remove_multicollinearity=True,
    multicollinearity_threshold=0.95,
    feature_selection=False,
    categorical_features=cat_cols or None,
    numeric_features=num_cols or None,
    use_gpu=False,
    n_jobs=1,              # ⬅️ penting: hindari crash loky/GaussianNB
    verbose=False,
)



# -------------------------------
# 2) Model selection & tuning
# -------------------------------
# Ambil 3 model terbaik, EXCLUDE Naive Bayes (nb) yang kemarin bikin error di VotingClassifier
base3 = compare_models(sort="AUC", n_select=2, exclude=["nb"], verbose=False)
tuned3 = [tune_model(m, optimize="AUC", verbose=False) for m in base3]
stacked = stack_models(
    estimator_list=tuned3,
    meta_model=LogisticRegression(max_iter=1000),
    optimize="AUC",
    verbose=False
)
final_model = finalize_model(stacked)

# -------------------------------
# 3) Evaluasi hold-out (test set) + threshold optimal
# -------------------------------
holdout_df = predict_model(final_model)  # holdout internal dari setup()
from sklearn.metrics import roc_auc_score, precision_recall_curve

auc_holdout = roc_auc_score(holdout_df[TARGET], holdout_df["prediction_score"])

prec, rec, thr = precision_recall_curve(holdout_df[TARGET], holdout_df["prediction_score"])
# thr punya panjang len(prec)-1; hitung F1 untuk tiap threshold
f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_thr = float(thr[int(np.argmax(f1))])

# metrik CV terakhir dari pull()
cv_tbl = pull().copy()
auc_cv_mean = float(cv_tbl.get("AUC").mean()) if "AUC" in cv_tbl else float("nan")
acc_cv_mean = float(cv_tbl.get("Accuracy").mean()) if "Accuracy" in cv_tbl else float("nan")

# siapkan metrics untuk MLflow (semua float, tanpa NaN)
metrics_raw = {
    "AUC_holdout": float(auc_holdout),
    "AUC_CV_mean": auc_cv_mean,
    "Accuracy_CV_mean": acc_cv_mean,
    "stacked": 1.0  # <— ganti blended_ok
}
metrics = {k: float(v) for k, v in metrics_raw.items() if isinstance(v, (int, float)) and np.isfinite(v)}
# -------------------------------
# -------------------------------
# 4) MLflow logging (tetap seperti punyamu)
# -------------------------------
with mlflow.start_run(run_name=f"pycaret_{int(time.time())}") as run:
    mlflow.log_metrics(metrics)
    feature_cols = [c for c in df.columns if c != TARGET]
    mlflow.log_dict({"feature_columns": feature_cols}, "feature_schema.json")

    save_path = "training/pycaret_pipeline"
    save_model(final_model, save_path)  # -> training/pycaret_pipeline.pkl
    mlflow.log_artifact(f"{save_path}.pkl", artifact_path="model")
    mlflow.set_tags({"ensemble": "stacking", "threshold_f1": f"{best_thr:.4f}", "sort_metric": "AUC"})

# -------------------------------
# 5) Buat bundle kompatibel FastAPI
# -------------------------------
import joblib, sklearn
from pycaret.classification import load_model

full_pipeline = load_model(save_path)
bundle = {
    "model": full_pipeline,
    "target_encoder": None,
    "feature_columns": feature_cols,
    "sklearn_version": sklearn.__version__,
    "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    "training_metrics": metrics_raw,
    "threshold": best_thr
}
os.makedirs("backend", exist_ok=True)
joblib.dump(bundle, "backend/model_bundle.joblib")
print("Saved bundle -> backend/model_bundle.joblib ; threshold(F1) =", best_thr)
