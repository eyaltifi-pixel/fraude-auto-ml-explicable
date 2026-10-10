import os, sys, subprocess
import joblib
import requests
import streamlit as st

@st.cache_resource
def start_api():
    """Lance l'API FastAPI en arrière-plan si elle ne tourne pas déjà."""
    try:
        requests.get("http://127.0.0.1:8000/health", timeout=2)
        return None
    except Exception:
        return subprocess.Popen([sys.executable, "-m", "uvicorn", "api:app", "--port", "8000"])

start_api()
import pandas as pd
import plotly.express as px
from core import *

st.set_page_config(page_title="Fraude auto : ML explicable", layout="wide")
st.title("Détection de fraude aux sinistres auto")

@st.cache_data
def get_data(file):
    return load_data(file) if file is not None else load_data()

# ---------- Barre latérale : source de données et cible ----------
with st.sidebar:
    st.header("Données")
    up = st.file_uploader("Importer un CSV (sinon : jeu de données par défaut)", type="csv")
    raw = get_data(up)
    cols = list(raw.columns)
    default_t = cols.index(DEFAULT_TARGET) if DEFAULT_TARGET in cols else len(cols) - 1
    target = st.selectbox("Variable cible", cols, index=default_t)
    values = sorted(raw[target].dropna().astype(str).unique())
    default_p = values.index(DEFAULT_POSITIVE) if DEFAULT_POSITIVE in values else 0
    positive = st.selectbox("Classe positive", values, index=default_p)

df = add_target(raw, target, positive)
num_all, cat_all = split_columns(df, target)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["1 · Données", "2 · Préparation", "3 · Modélisation", "4 · Évaluation", "6 · API"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes", f"{len(df):,}")
    c2.metric("Colonnes", df.shape[1] - 1)
    c3.metric("Taux de classe positive", f"{df['target'].mean():.1%}")

    st.subheader("Aperçu")
    st.dataframe(df.head(20), width="stretch")
    st.subheader("Statistiques descriptives")
    st.dataframe(df[num_all].describe().T, width="stretch")

    left, right = st.columns(2)
    with left:
        st.subheader("Valeurs manquantes")
        miss = (df.isna().mean() * 100).round(1)
        miss = miss[miss > 0].sort_values(ascending=False)
        st.dataframe(miss.rename("% manquant")) if len(miss) else st.write("Aucune")
    with right:
        st.subheader("Répartition des classes")
        st.plotly_chart(px.pie(df, names=target), width="stretch")

    st.subheader("Distribution d'une variable")
    var = st.selectbox("Variable numérique", num_all)
    st.plotly_chart(px.histogram(df, x=var, color=target, barmode="overlay",
                                 opacity=0.6, marginal="box"), width="stretch")

    st.subheader("Corrélations")
    st.plotly_chart(px.imshow(df[num_all + ["target"]].corr().round(2), text_auto=True, aspect="auto",
                              color_continuous_scale="RdBu_r", zmin=-1, zmax=1),
                    width="stretch")

    st.subheader("Taux de la classe positive par variable")
    col = st.selectbox("Variable catégorielle", cat_all)
    t = (df.assign(**{col: df[col].fillna("Manquant").astype(str)})
           .groupby(col)["target"].agg(taux="mean", n="size").reset_index()
           .sort_values("taux", ascending=False))
    t["taux"] = (t["taux"] * 100).round(1)
    fig = px.bar(t, x=col, y="taux", text="n")
    fig.add_hline(y=df["target"].mean() * 100, line_dash="dash", annotation_text="taux global")
    st.plotly_chart(fig, width="stretch")

with tab2:
    df2 = engineer_features(df)
    num2, cat2 = split_columns(df2, target)
    options = num2 + cat2

    st.subheader("Variables explicatives")
    feats = st.multiselect("Variables à utiliser", options,
                           default=[c for c in options if c not in REDUNDANT])
    if any(c in feats for c in REDUNDANT):
        st.warning("`total_claim_amount` est la somme des trois autres montants : "
                   "la garder rend les coefficients instables.")
    sens = [c for c in SENSITIVE if c in feats]
    if sens:
        st.caption("Variables sensibles incluses (à surveiller pour l'équité) : " + ", ".join(sens))
    if not feats:
        st.stop()

    st.subheader("Traitements")
    a, b, c = st.columns(3)
    num_imp = a.selectbox("Imputation numérique", ["median", "mean", "most_frequent"])
    scaler = b.selectbox("Normalisation (régression logistique, KNN)", list(SCALERS))
    test_size = c.slider("Part du jeu de test", 0.10, 0.40, 0.20, 0.05)
    seed = st.number_input("Graine aléatoire", value=42, step=1)
    st.caption("Valeurs catégorielles manquantes → modalité « Manquant » (informative dans ce jeu de données). "
               "Encodage : one-hot.")

    num_f = [c for c in feats if c in num2]
    cat_f = [c for c in feats if c in cat2]
    d2 = df2.copy()
    d2[cat_f] = d2[cat_f].astype(object).fillna("Manquant")

    X_train, X_test, y_train, y_test = make_split(d2, feats, test_size, int(seed))
    prep = make_preprocessor(num_f, cat_f, scaler="Aucune", num_impute=num_imp).fit(X_train)
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Train", f"{len(X_train)} lignes")
    m2.metric("Test", f"{len(X_test)} lignes")
    m3.metric("Positifs train / test", f"{y_train.mean():.1%} / {y_test.mean():.1%}")
    m4.metric("Colonnes après encodage", len(names))

    st.subheader("Aperçu des données transformées")
    st.dataframe(pd.DataFrame(prep.transform(X_train.head(5)), columns=names), width="stretch")

    st.session_state["prep"] = dict(X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
                                    num=num_f, cat=cat_f, features=feats, scaler=scaler,
                                    num_imp=num_imp, target=target, positive=positive)
    st.success("Préparation enregistrée : elle sera utilisée par l'onglet Modélisation.")

with tab3:
    P = st.session_state["prep"]
    Xtr, ytr = P["X_train"], P["y_train"]
    pos_w = float((ytr == 0).sum() / max((ytr == 1).sum(), 1))

    st.subheader("Choix des modèles")
    chosen = st.multiselect("Modèles à entraîner", MODEL_NAMES, default=MODEL_NAMES)
    mode = st.radio("Hyperparamètres", ["Réglage manuel", "Optimisation automatique"], horizontal=True)

    manual = {}
    if mode == "Réglage manuel":
        for name in chosen:
            with st.expander(f"Hyperparamètres : {name}"):
                manual[name] = {}
                for p, (label, rng, default) in HP_UI[name].items():
                    key = f"hp_{name}_{p}"
                    if isinstance(rng, list):
                        manual[name][p] = st.select_slider(label, options=rng, value=default, key=key)
                    elif isinstance(default, float):
                        manual[name][p] = st.slider(label, float(rng[0]), float(rng[1]), default, key=key)
                    else:
                        manual[name][p] = st.slider(label, rng[0], rng[1], default, key=key)
        n_iter = 15
    else:
        n_iter = st.slider("Nombre de combinaisons testées par modèle", 5, 40, 15)
        st.caption("Recherche aléatoire, critère : PR-AUC en validation croisée sur le jeu d'entraînement.")
    folds = st.slider("Plis de validation croisée", 3, 10, 5)

    if st.button("Entraîner et comparer", type="primary", key="train"):
        if not chosen:
            st.error("Choisis au moins un modèle.")
        else:
            rows, fitted, best_params = [], {}, {}
            bar = st.progress(0.0)
            for i, name in enumerate(chosen):
                with st.spinner(f"{name}…"):
                    if mode == "Optimisation automatique":
                        s = tune(name, P, Xtr, ytr, n_iter=n_iter, folds=folds, pos_w=pos_w)
                        fitted[name] = s.best_estimator_
                        best_params[name] = {k.replace("clf__", ""): (round(v, 4) if isinstance(v, float) else v)
                                             for k, v in s.best_params_.items()}
                        m = evaluate_cv(s.best_estimator_, Xtr, ytr, folds)
                    else:
                        pipe = build_pipe(name, P, manual[name], pos_w)
                        m = evaluate_cv(pipe, Xtr, ytr, folds)
                        fitted[name] = pipe.fit(Xtr, ytr)
                        best_params[name] = manual[name]
                rows.append({"Modèle": name, "ROC-AUC": m["roc_auc"], "PR-AUC": m["average_precision"],
                             "F1": m["f1"], "Rappel": m["recall"], "Précision": m["precision"]})
                bar.progress((i + 1) / len(chosen))
            table = pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).reset_index(drop=True)
            st.session_state["models"] = dict(fitted=fitted, table=table, params=best_params, mode=mode, sig=make_sig(P))
            joblib.dump({"models": fitted, "features": P["features"]}, "bundle.tmp")
            os.replace("bundle.tmp", "bundle.joblib")

    if "models" in st.session_state:
        R = st.session_state["models"]
        if R.get("sig") != make_sig(P):
            st.warning("Les réglages de préparation ont changé depuis cet entraînement : relance « Entraîner et comparer ».")
        st.subheader("Tableau comparatif (validation croisée sur le train)")
        st.dataframe(R["table"].round(3), width="stretch", hide_index=True)
        fig = px.bar(R["table"], x="Modèle", y="PR-AUC", text=R["table"]["PR-AUC"].round(3))
        fig.add_hline(y=float(ytr.mean()), line_dash="dash", annotation_text="hasard (taux de positifs)")
        st.plotly_chart(fig, width="stretch")
        with st.expander("Hyperparamètres retenus"):
            st.json(R["params"])
        st.caption("Écarts de 0,01 à 0,02 entre modèles : à considérer comme du bruit avec un petit jeu de données. "
                   "Le jeu de test n'est utilisé qu'à l'onglet Évaluation.")

import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import (roc_auc_score, average_precision_score, f1_score, recall_score,
                             precision_score, roc_curve, precision_recall_curve, confusion_matrix)

with tab4:
    if "models" not in st.session_state:
        st.info("Entraîne d'abord les modèles dans l'onglet 3 · Modélisation.")
    elif st.session_state["models"].get("sig") != make_sig(st.session_state["prep"]):
        st.warning("Les réglages de préparation ont changé depuis l'entraînement : "
                   "relance « Entraîner et comparer » dans l'onglet 3 avant d'évaluer.")
    else:
        P, R = st.session_state["prep"], st.session_state["models"]
        Xte, yte = P["X_test"], P["y_test"].to_numpy()
        proba = {n: m.predict_proba(Xte)[:, 1] for n, m in R["fitted"].items()}

        thr = st.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05, key="thr")

        rows = []
        for n, p in proba.items():
            pred = (p >= thr).astype(int)
            rows.append({"Modèle": n, "ROC-AUC": roc_auc_score(yte, p), "PR-AUC": average_precision_score(yte, p),
                         "Précision": precision_score(yte, pred, zero_division=0),
                         "Rappel": recall_score(yte, pred, zero_division=0),
                         "F1": f1_score(yte, pred, zero_division=0)})
        st.subheader(f"Métriques sur le jeu de test (seuil = {thr:.2f})")
        st.dataframe(pd.DataFrame(rows).sort_values("PR-AUC", ascending=False).round(3),
                     hide_index=True, width="stretch")
        st.caption(f"Le test contient {int(yte.sum())} positifs sur {len(yte)} : un seul dossier de plus ou de moins "
                   "déplace le rappel de plusieurs points. ROC-AUC et PR-AUC ne dépendent pas du seuil.")

        c1, c2 = st.columns(2)
        fig = go.Figure()
        for n, p in proba.items():
            fpr, tpr, _ = roc_curve(yte, p)
            fig.add_scatter(x=fpr, y=tpr, name=f"{n} ({roc_auc_score(yte, p):.2f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], line=dict(dash="dash"), name="Hasard")
        fig.update_layout(title="Courbes ROC", xaxis_title="Taux de faux positifs", yaxis_title="Rappel")
        c1.plotly_chart(fig, width="stretch")

        fig = go.Figure()
        for n, p in proba.items():
            pr, rc, _ = precision_recall_curve(yte, p)
            fig.add_scatter(x=rc, y=pr, name=f"{n} ({average_precision_score(yte, p):.2f})")
        fig.add_hline(y=float(yte.mean()), line_dash="dash", annotation_text="hasard")
        fig.update_layout(title="Courbes précision-rappel", xaxis_title="Rappel", yaxis_title="Précision")
        c2.plotly_chart(fig, width="stretch")

        st.subheader("Analyse d'un modèle")
        name = st.selectbox("Modèle", list(proba), key="eval_model")
        p = proba[name]
        pred = (p >= thr).astype(int)
        a, b = st.columns(2)
        cm = confusion_matrix(yte, pred, labels=[0, 1])
        a.plotly_chart(px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                                 x=["Prédit négatif", "Prédit positif"], y=["Réel négatif", "Réel positif"],
                                 title=f"Matrice de confusion (seuil {thr:.2f})"), width="stretch")
        ths = np.linspace(0.02, 0.98, 49)
        curve = pd.DataFrame({
            "Seuil": ths,
            "Précision": [precision_score(yte, (p >= t).astype(int), zero_division=0) for t in ths],
            "Rappel": [recall_score(yte, (p >= t).astype(int), zero_division=0) for t in ths],
            "F1": [f1_score(yte, (p >= t).astype(int), zero_division=0) for t in ths]})
        fig = px.line(curve, x="Seuil", y=["Précision", "Rappel", "F1"], title="Effet du seuil")
        fig.add_vline(x=thr, line_dash="dash")
        b.plotly_chart(fig, width="stretch")
        st.caption("Baisser le seuil détecte plus de fraudes (rappel ↑) mais génère plus de fausses alertes (précision ↓). "
                   "Le seuil doit être choisi selon le coût d'une enquête inutile face à une fraude manquée, "
                   "et surtout pas en regardant ce jeu de test.")

import requests

with tab5:
    st.subheader("Prédiction via l'API FastAPI")
    api_url = st.text_input("URL de l'API", "http://127.0.0.1:8000")

    if "prep" not in st.session_state:
        st.info("Prépare d'abord les données dans l'onglet 2 · Préparation.")
        st.stop()
    try:
        info = requests.get(f"{api_url}/models", timeout=5)
        info.raise_for_status()
        info = info.json()
    except Exception as e:
        st.error(f"API injoignable ou sans modèle entraîné : {e}")
        st.stop()

    P = st.session_state["prep"]
    feats = info["variables_attendues"]
    if feats != list(P["features"]):
        st.warning("Les variables de l'API ne correspondent plus à la préparation actuelle : "
                   "relance « Entraîner et comparer » dans l'onglet 3.")
        st.stop()

    Xtr, Xte, yte = P["X_train"], P["X_test"], P["y_test"]
    st.caption("Le formulaire est pré-rempli avec un dossier du jeu de test ; tu peux modifier les valeurs.")
    pos = int(st.number_input("Dossier du jeu de test (n° de ligne)", 0, len(Xte) - 1, 0))
    row = Xte.iloc[pos]
    st.write(f"Vérité terrain de ce dossier : **{'positif' if yte.iloc[pos] == 1 else 'négatif'}**")

    values = {}
    cols = st.columns(3)
    for i, f in enumerate(feats):
        box, key = cols[i % 3], f"api_{f}_{pos}"
        if f in P["num"]:
            v = row[f]
            v = Xtr[f].median() if pd.isna(v) else v
            if pd.api.types.is_integer_dtype(Xtr[f]):
                values[f] = box.number_input(f, value=int(v), step=1, key=key)
            else:
                values[f] = box.number_input(f, value=float(v), format="%.2f", key=key)
        else:
            opts = sorted(Xtr[f].astype(str).unique())
            cur = str(row[f])
            values[f] = box.selectbox(f, opts, index=opts.index(cur) if cur in opts else 0, key=key)

    c1, c2 = st.columns(2)
    names = info["modeles"]
    modele = c1.selectbox("Modèle", names, index=names.index("XGBoost") if "XGBoost" in names else 0, key="api_model")
    seuil = c2.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05, key="api_thr")

    if st.button("Prédire via l'API", type="primary", key="api_go"):
        r = requests.post(f"{api_url}/predict",
                          json={"modele": modele, "seuil": seuil, "variables": values}, timeout=30)
        if r.ok:
            out = r.json()
            m1, m2 = st.columns(2)
            m1.metric("Probabilité", f"{out['probabilite_fraude']:.1%}")
            m2.metric("Décision", out["decision"])
            st.progress(min(float(out["probabilite_fraude"]), 1.0))
        else:
            st.error(f"Erreur {r.status_code} : {r.json().get('detail', r.text)}")
