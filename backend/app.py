from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os, sys, traceback
import joblib, pandas as pd, numpy as np, sklearn
from typing import Optional, List

# --- Lokasi bundle di container (Cloud Run) ---
BUNDLE_PATH = os.getenv("BUNDLE_PATH", "/app/model_bundle.joblib")

# --- Helper: coba ambil daftar kolom dari ColumnTransformer bernama 'preprocessor' ---
def _extract_feature_columns_from_pipeline(pipe) -> Optional[List[str]]:
    try:
        pre = None
        if hasattr(pipe, "named_steps") and "preprocessor" in pipe.named_steps:
            pre = pipe.named_steps["preprocessor"]
        else:
            try:
                pre = pipe["preprocessor"]
            except Exception:
                pre = None
        if pre is None:
            return None
        cols: List[str] = []
        for name, trans, cols_sel in getattr(pre, "transformers_", []):
            if name == "remainder":
                continue
            if isinstance(cols_sel, (list, tuple, np.ndarray, pd.Index)):
                cols.extend(list(cols_sel))
        return cols or None
    except Exception:
        return None

# --- Load bundle dengan proteksi error ---
bundle_load_error = None
try:
    loaded = joblib.load(BUNDLE_PATH)
except Exception as e:
    bundle_load_error = repr(e)
    traceback.print_exc()
    loaded = {"model": None, "feature_columns": [], "error": bundle_load_error}

# --- Normalisasi objek dari bundle ---
if isinstance(loaded, dict):
    model = loaded.get("model")
    feature_columns = loaded.get("feature_columns") or (model and _extract_feature_columns_from_pipeline(model))
    le_y = loaded.get("target_encoder")  # bisa None
    sklearn_trained = loaded.get("sklearn_version", "unknown")
    created_at = loaded.get("created_at")
    training_metrics = loaded.get("training_metrics", {})
    threshold = float(loaded.get("threshold", os.getenv("THRESHOLD", "0.5")))
else:
    # Jika file adalah Pipeline langsung (kurang ideal), coba ambil kolom dari preprocessor
    model = loaded
    feature_columns = _extract_feature_columns_from_pipeline(model)
    le_y, sklearn_trained, created_at, training_metrics = None, "unknown", None, {}
    threshold = float(os.getenv("THRESHOLD", "0.5"))

# --- Validasi minimum ---
if not feature_columns:
    # Masih boleh start, tapi tandai error agar kelihatan di /health
    bundle_load_error = bundle_load_error or "feature_columns tidak tersedia di bundle dan tidak dapat diekstrak dari pipeline."

# --- FastAPI app ---
app = FastAPI(title="Bank Churn API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

# --- Schemas (Pydantic v2) ---
class ChurnRequest(BaseModel):
    credit_score: float
    country: str
    gender: str
    age: int
    tenure: int
    balance: float
    products_number: int
    credit_card: int
    active_member: int
    estimated_salary: float

class BatchRequest(BaseModel):
    items: list[ChurnRequest]

# --- Utils ---
def align_df(df: pd.DataFrame) -> pd.DataFrame:
    if not feature_columns:
        return df
    missing = [c for c in feature_columns if c not in df.columns]
    for c in missing:
        df[c] = pd.NA
    return df[feature_columns]

def proba_of_churn(m, X: pd.DataFrame) -> np.ndarray:
    """Return probability for positive class."""
    if m is None:
        raise RuntimeError("Model not loaded.")
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    # Fallback untuk estimator tanpa predict_proba (jarang di PyCaret untuk klasifikasi biner)
    if hasattr(m, "decision_function"):
        raw = m.decision_function(X)
        if raw.ndim == 1:
            return 1.0 / (1.0 + np.exp(-raw))  # sigmoid
        # multiclass -> ambil kolom indeks 1 (heuristik)
        ex = np.exp(raw - raw.max(axis=1, keepdims=True))
        sm = ex / ex.sum(axis=1, keepdims=True)
        return sm[:, 1]
    # Terakhir: pakai prediksi 0/1 sebagai probabilitas kasar
    return m.predict(X).astype(float)

# --- Endpoints ---
@app.get("/health")
def health():
    info = {
        "status": "ok" if model is not None and not bundle_load_error else "degraded",
        "loaded": model is not None,
        "bundle_path": BUNDLE_PATH,
        "sklearn_runtime": sklearn.__version__,
        "sklearn_trained": sklearn_trained,
        "created_at": created_at,
        "training_metrics": training_metrics,
        "threshold": threshold,
        "feature_count": len(feature_columns or []),
    }
    if bundle_load_error:
        info["error"] = bundle_load_error
    return info

@app.get("/features")
def features():
    return {"features": feature_columns or []}

@app.post("/predict")
def predict_one(item: ChurnRequest):
    df_new = pd.DataFrame([item.model_dump()])
    df_new = align_df(df_new)
    prob = float(proba_of_churn(model, df_new)[0])
    pred = int(prob >= threshold)
    label = le_y.inverse_transform([pred])[0] if le_y is not None else pred
    return {"prediction": pred, "label": label, "probability": prob}

@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    df_new = pd.DataFrame([i.model_dump() for i in req.items])
    df_new = align_df(df_new)
    probs = proba_of_churn(model, df_new).tolist()
    preds = [int(p >= threshold) for p in probs]
    labels = le_y.inverse_transform(preds).tolist() if le_y is not None else preds
    return {"predictions": preds, "labels": labels, "probabilities": probs}
