import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler

URL = "https://raw.githubusercontent.com/mwitiderrick/insurancedata/master/insurance_claims.csv"
DEFAULT_TARGET, DEFAULT_POSITIVE = "fraud_reported", "Y"
ID_COLS = ["policy_number", "insured_zip"]
RAW_DROP = ["incident_location", "policy_bind_date", "incident_date"]
REDUNDANT = ["total_claim_amount"]          # somme des trois autres montants
SENSITIVE = ["insured_sex", "age", "insured_education_level", "insured_occupation", "insured_relationship"]
SCALERS = {"Standard (z-score)": StandardScaler, "Min-Max": MinMaxScaler, "Aucune": None}

def load_data(source=URL):
    return pd.read_csv(source, na_values=["?"]).dropna(axis=1, how="all")

def add_target(df, target, positive):
    df = df.copy()
    df["target"] = (df[target].astype(str) == str(positive)).astype(int)
    return df

def engineer_features(df):
    """Variables dérivées des dates (uniquement si les colonnes existent)."""
    df = df.copy()
    if {"incident_date", "policy_bind_date"} <= set(df.columns):
        d = pd.to_datetime(df["incident_date"], errors="coerce")
        b = pd.to_datetime(df["policy_bind_date"], errors="coerce")
        df["policy_age_days"] = (d - b).dt.days
        df["incident_dow"] = d.dt.dayofweek
    return df

def split_columns(df, target):
    """Colonnes numériques / catégorielles utilisables (hors cible, identifiants, dates brutes)."""
    drop = set(ID_COLS + RAW_DROP + ["target", target])
    drop |= {c for c in df.columns if df[c].nunique() == len(df)}      # identifiants strictement uniques
    num = [c for c in df.select_dtypes("number").columns if c not in drop]
    cat = [c for c in df.select_dtypes(exclude="number").columns if c not in drop]
    return num, cat

def make_split(df, features, test_size=0.2, seed=42):
    return train_test_split(df[features], df["target"], test_size=test_size,
                            stratify=df["target"], random_state=seed)

def make_preprocessor(num, cat, scaler="Standard (z-score)", num_impute="median"):
    steps = [("imp", SimpleImputer(strategy=num_impute))]
    if SCALERS[scaler] is not None:
        steps.append(("sc", SCALERS[scaler]()))
    return ColumnTransformer([
        ("num", Pipeline(steps), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat)])


# ======================= Module 3 : modèles =======================
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_validate, RandomizedSearchCV
from xgboost import XGBClassifier
from scipy.stats import randint, loguniform, uniform

MODEL_NAMES = ["Régression logistique", "Arbre de décision", "KNN", "Random Forest", "XGBoost"]
SCALED = {"Régression logistique", "KNN"}          # modèles sensibles à l'échelle

# Réglages manuels : (libellé, plage ou liste de valeurs, défaut)
HP_UI = {
    "Régression logistique": {"C": ("C (inverse de la régularisation)", [0.001, 0.01, 0.1, 1.0, 10.0, 100.0], 1.0)},
    "Arbre de décision": {"max_depth": ("Profondeur max", (2, 12), 3),
                          "min_samples_leaf": ("Taille min. d'une feuille", (1, 40), 6)},
    "KNN": {"n_neighbors": ("Nombre de voisins", (3, 51), 15)},
    "Random Forest": {"n_estimators": ("Nombre d'arbres", (100, 600), 300),
                      "max_depth": ("Profondeur max", (3, 15), 8),
                      "min_samples_leaf": ("Taille min. d'une feuille", (1, 15), 3)},
    "XGBoost": {"n_estimators": ("Nombre d'arbres", (50, 500), 150),
                "max_depth": ("Profondeur max", (2, 8), 3),
                "learning_rate": ("Taux d'apprentissage", (0.01, 0.30), 0.05)},
}

PARAM_GRIDS = {
    "Régression logistique": {"clf__C": loguniform(1e-3, 1e2)},
    "Arbre de décision": {"clf__max_depth": randint(2, 10), "clf__min_samples_leaf": randint(5, 40)},
    "KNN": {"clf__n_neighbors": randint(3, 51)},
    "Random Forest": {"clf__n_estimators": randint(200, 600), "clf__max_depth": randint(3, 12),
                      "clf__min_samples_leaf": randint(1, 15), "clf__max_features": ["sqrt", 0.3, 0.5]},
    "XGBoost": {"clf__n_estimators": randint(100, 500), "clf__max_depth": randint(2, 6),
                "clf__learning_rate": loguniform(0.01, 0.2), "clf__subsample": uniform(0.6, 0.4),
                "clf__colsample_bytree": uniform(0.5, 0.5), "clf__min_child_weight": randint(1, 10)},
}

def make_estimator(name, params, pos_w=1.0):
    params = dict(params or {})
    if name == "Régression logistique":
        return LogisticRegression(max_iter=2000, class_weight="balanced", **params)
    if name == "Arbre de décision":
        return DecisionTreeClassifier(class_weight="balanced", random_state=42, **params)
    if name == "KNN":
        return KNeighborsClassifier(**params)
    if name == "Random Forest":
        return RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1, **params)
    return XGBClassifier(scale_pos_weight=pos_w, eval_metric="logloss", random_state=42, **params)

def build_pipe(name, cfg, params=None, pos_w=1.0):
    scaler = cfg["scaler"] if name in SCALED else "Aucune"
    prep = make_preprocessor(cfg["num"], cfg["cat"], scaler=scaler, num_impute=cfg["num_imp"])
    return Pipeline([("prep", prep), ("clf", make_estimator(name, params, pos_w))])

SCORING = ["roc_auc", "average_precision", "f1", "recall", "precision"]

def evaluate_cv(pipe, X, y, folds=5, seed=42):
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    r = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, n_jobs=-1)
    return {k.replace("test_", ""): float(r[k].mean()) for k in r if k.startswith("test_")}

def tune(name, cfg, X, y, n_iter=15, folds=5, seed=42, pos_w=1.0):
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    s = RandomizedSearchCV(build_pipe(name, cfg, None, pos_w), PARAM_GRIDS[name], n_iter=n_iter,
                           scoring="average_precision", cv=cv, n_jobs=-1, random_state=seed)
    return s.fit(X, y)


def make_sig(P):
    """Signature de la préparation : permet de détecter des modèles entraînés avec d'autres réglages."""
    return (tuple(P["features"]), tuple(P["X_test"].index), P["scaler"], P["num_imp"])
