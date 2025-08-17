# app.py (bagian atas)
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib, pandas as pd, sklearn, numpy as np
import os
from typing import Optional, List, Union

BUNDLE_PATH = os.environ.get("BUNDLE_PATH", "model_bundle.joblib")  # ganti ke 'best_model.pkl' kalau itu filenya

def _extract_feature_columns_from_pipeline(pipe) -> Optional[List[str]]:
    try:
        # cari step 'preprocessor' di pipeline
        pre = None
        if hasattr(pipe, "named_steps") and "preprocessor" in pipe.named_steps:
            pre = pipe.named_steps["preprocessor"]
        elif hasattr(pipe, "__getitem__"):
            try:
                pre = pipe["preprocessor"]
            except Exception:
                pre = None
        if pre is None:
            return None

        cols: List[str] = []
        # pre.transformers_ berisi (name, transformer, columns_yang_dipakai_saat_fit)
        for name, trans, cols_sel in getattr(pre, "transformers_", []):
            if name == "remainder":
                continue
            if isinstance(cols_sel, (list, tuple, np.ndarray, pd.Index)):
                cols.extend(list(cols_sel))
        return cols or None
    except Exception:
        return None

loaded = joblib.load(BUNDLE_PATH)

# Normalisasi ke objek2 yang kita pakai di endpoint
if isinstance(loaded, dict):
    model = loaded["model"]
    feature_columns = loaded.get("feature_columns") or _extract_feature_columns_from_pipeline(model)
    le_y = loaded.get("target_encoder", None)
    sklearn_trained = loaded.get("sklearn_version", "unknown")
else:
    # file berisi Pipeline langsung
    model = loaded
    feature_columns = _extract_feature_columns_from_pipeline(model)
    le_y = None
    sklearn_trained = "unknown"

if feature_columns is None:
    raise RuntimeError(
        "Tidak bisa menentukan daftar feature_columns. "
        "Solusi cepat: simpan ulang model sebagai bundle yang menyertakan feature_columns, "
        "atau pastikan pipeline punya step 'preprocessor' ColumnTransformer yang fitted."
    )

app = FastAPI(title="Bank Churn API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"]
)

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

def align_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in feature_columns if c not in df.columns]
    for c in missing:
        df[c] = pd.NA
    return df[feature_columns]

@app.get("/ping")
def ping():
    return {"status": "ok"}

@app.get("/features")
def features():
    return {"features": feature_columns}

@app.get("/metadata")
def metadata():
    return {
        "sklearn_version_runtime": sklearn.__version__,
        "sklearn_version_trained": sklearn_trained,
        "has_label_encoder": le_y is not None,
        "bundle_path": BUNDLE_PATH,
        "is_bundle_dict": isinstance(loaded, dict)
    }

@app.post("/predict")
def predict_one(item: ChurnRequest):
    df_new = pd.DataFrame([item.model_dump()])
    df_new = align_df(df_new)
    proba = float(model.predict_proba(df_new)[:, 1][0])
    pred = int(model.predict(df_new)[0])
    label = le_y.inverse_transform([pred])[0] if le_y is not None else pred
    return {"prediction": pred, "label": label, "probability": proba}

@app.post("/predict_batch")
def predict_batch(req: BatchRequest):
    df_new = pd.DataFrame([i.model_dump() for i in req.items])
    df_new = align_df(df_new)
    probs = model.predict_proba(df_new)[:, 1].tolist()
    preds = model.predict(df_new).tolist()
    labels = le_y.inverse_transform(preds).tolist() if le_y is not None else preds
    return {"predictions": preds, "labels": labels, "probabilities": probs}
