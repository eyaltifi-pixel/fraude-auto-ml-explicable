import os
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

BUNDLE_PATH = "bundle.joblib"
_cache = {"mtime": None, "bundle": None}

def get_bundle():
    """Recharge le bundle s'il a changé (réentraînement depuis l'application)."""
    if not os.path.exists(BUNDLE_PATH):
        raise HTTPException(503, "Aucun modèle entraîné : entraîne d'abord les modèles dans l'application.")
    m = os.path.getmtime(BUNDLE_PATH)
    if _cache["mtime"] != m:
        _cache.update(mtime=m, bundle=joblib.load(BUNDLE_PATH))
    return _cache["bundle"]

app = FastAPI(
    title="API de détection de fraude auto",
    description="Prédit la probabilité qu'un dossier de sinistre automobile soit frauduleux.",
    version="1.1",
)

class PredictRequest(BaseModel):
    modele: str = Field("XGBoost", description="Nom du modèle (voir GET /models)")
    seuil: float = Field(0.5, ge=0, le=1, description="Seuil de décision")
    variables: dict = Field(..., description="Valeurs des variables explicatives du dossier")

class PredictResponse(BaseModel):
    modele: str
    probabilite_fraude: float
    seuil: float
    decision: str

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/models")
def list_models():
    b = get_bundle()
    return {"modeles": list(b["models"]), "variables_attendues": b["features"]}

@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    b = get_bundle()
    models, features = b["models"], b["features"]
    if req.modele not in models:
        raise HTTPException(404, f"Modèle inconnu : {req.modele}")
    missing = [f for f in features if f not in req.variables]
    if missing:
        raise HTTPException(422, f"Variables manquantes : {missing}")
    X = pd.DataFrame([req.variables])[features]
    p = float(models[req.modele].predict_proba(X)[0, 1])
    return PredictResponse(
        modele=req.modele,
        probabilite_fraude=round(p, 4),
        seuil=req.seuil,
        decision="À investiguer" if p >= req.seuil else "Traitement standard",
    )


# ======================= Module 6 : API d'interprétation =======================
from core import SENSITIVE
from xai_core import explain_instance, SHAP_MODELS


class ExplainRequest(BaseModel):
    modele: str = Field("XGBoost", description="Nom du modèle (voir GET /models)")
    seuil: float = Field(0.5, ge=0, le=1, description="Seuil de décision")
    top_k: int = Field(8, ge=1, le=30, description="Nombre de variables renvoyées")
    variables: dict = Field(..., description="Valeurs des variables explicatives du dossier")


@app.post("/explain", summary="Explique la prédiction pour un dossier (SHAP)")
def explain(req: ExplainRequest):
    b = get_bundle()
    if req.modele not in b["models"]:
        raise HTTPException(404, f"Modèle inconnu : {req.modele}")
    if req.modele not in SHAP_MODELS:
        raise HTTPException(400, f"Explication SHAP non disponible pour {req.modele}. "
                                 f"Modèles expliqués : {', '.join(SHAP_MODELS)}")
    if "background" not in b:
        raise HTTPException(503, "Modèles exportés par une ancienne version : relance « Entraîner et comparer » dans l'application.")
    missing = [f for f in b["features"] if f not in req.variables]
    if missing:
        raise HTTPException(422, f"Variables manquantes : {missing}")
    return explain_instance(b, req.modele, req.variables, req.seuil, req.top_k, SENSITIVE)
