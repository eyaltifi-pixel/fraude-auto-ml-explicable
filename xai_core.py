"""Explicabilité (module 5) : fonctions sans Streamlit.
SHAP et LIME sont importés dans les fonctions qui les utilisent."""
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

TREE_MODELS = ("Arbre de décision", "Random Forest", "XGBoost")
SHAP_MODELS = ("Régression logistique",) + TREE_MODELS      # KNN : trop coûteux avec SHAP
LOG_ODDS = {"Régression logistique", "XGBoost"}              # les autres sortent des probabilités


def scale_label(name):
    return "log-odds (échelle logistique)" if name in LOG_ODDS else "probabilité"


def split_names(pipe, cat):
    """Noms des colonnes transformées et variable d'origine de chaque colonne."""
    raw = list(pipe.named_steps["prep"].get_feature_names_out())    # 'num__age', 'cat__incident_severity_Major Damage'
    cats = sorted(cat, key=len, reverse=True)
    names, origin = [], []
    for r in raw:
        kind, rest = r.split("__", 1)
        names.append(rest)
        origin.append(rest if kind == "num" else next(c for c in cats if rest.startswith(c + "_")))
    return names, origin


def group_by_origin(vals, origin):
    """Somme des contributions des colonnes one-hot d'une même variable (l'additivité SHAP est conservée)."""
    return pd.DataFrame(np.asarray(vals)).T.groupby(np.asarray(origin), sort=False).sum().T


def intrinsic_importance(name, pipe, cat):
    """Importance propre au modèle. Retourne (DataFrame[valeur, origine], libellé) ou (None, message)."""
    clf = pipe.named_steps["clf"]
    names, origin = split_names(pipe, cat)
    if name == "Régression logistique":
        df = pd.DataFrame({"valeur": clf.coef_[0], "origine": origin}, index=names)
        df = df.reindex(df["valeur"].abs().sort_values(ascending=False).index)
        return df, "Coefficients (variables numériques normalisées, catégories en one-hot) : le signe indique le sens"
    if hasattr(clf, "feature_importances_"):
        s = pd.Series(clf.feature_importances_, index=origin).groupby(level=0).sum().sort_values(ascending=False)
        df = pd.DataFrame({"valeur": s.to_numpy(), "origine": s.index}, index=s.index)
        kind = "Gain moyen des séparations, somme par variable" if name == "XGBoost" \
            else "Importance par impureté (baisse moyenne d'impureté), somme par variable"
        return df, kind
    return None, "Ce modèle n'a pas d'importance interne (KNN) : voir l'importance par permutation."


def permutation_table(fitted, X, y, n_repeats=5, seed=42):
    """Importance par permutation (baisse de PR-AUC), variables d'origine x modèles : comparable entre modèles."""
    out = {}
    for name, pipe in fitted.items():
        r = permutation_importance(pipe, X, y, scoring="average_precision",
                                   n_repeats=n_repeats, random_state=seed, n_jobs=-1)
        out[name] = pd.Series(r.importances_mean, index=X.columns)
    return pd.DataFrame(out)


def shap_values(name, pipe, Xbg, Xexp, cat):
    """Valeurs SHAP sur les colonnes transformées pour les lignes de Xexp."""
    import shap
    prep, clf = pipe.named_steps["prep"], pipe.named_steps["clf"]
    names, origin = split_names(pipe, cat)
    Tbg = pd.DataFrame(prep.transform(Xbg), columns=names)
    Tex = pd.DataFrame(prep.transform(Xexp), columns=names, index=Xexp.index)
    A, B = Tex.to_numpy(dtype=float), Tbg.to_numpy(dtype=float)
    if name == "Régression logistique":
        ex = shap.LinearExplainer(clf, B)(A)
        vals, base = ex.values, ex.base_values
    elif name == "XGBoost":
        try:
            ex = shap.TreeExplainer(clf)(A)
            vals, base = ex.values, ex.base_values
        except Exception:                      # secours : TreeSHAP natif de XGBoost
            import xgboost as xgb
            c = clf.get_booster().predict(xgb.DMatrix(A), pred_contribs=True)
            vals, base = c[:, :-1], c[:, -1]
    elif name in TREE_MODELS:
        ex = shap.TreeExplainer(clf)(A, check_additivity=False)
        vals, base = ex.values, ex.base_values
    else:
        raise ValueError(f"SHAP non calculé pour {name}")
    vals, base = np.asarray(vals, dtype=float), np.asarray(base, dtype=float)
    if vals.ndim == 3:                         # arbres sklearn : (n, variables, classes) -> classe positive
        vals, base = vals[:, :, 1], base[..., 1]
    base = np.broadcast_to(base.reshape(-1) if base.ndim else base, (len(vals),)).copy()
    return dict(vals=vals, base=base, T=Tex, names=names, origin=origin)


# ======================= Explications locales (étape 2) =======================

def display_data(S, Xexp, num):
    """Valeurs affichées dans le waterfall : variables numériques d'origine (non normalisées)."""
    D = S["T"].copy()
    for c in num:
        if c in D.columns:
            raw = pd.to_numeric(Xexp[c], errors="coerce")
            D[c] = raw.where(raw.notna(), D[c])
    return D.to_numpy(dtype=float)


class LimeRaw:
    """LIME sur les variables d'origine (catégories lisibles), en passant par le pipeline complet."""

    def __init__(self, Xtr, num, cat, seed=42):
        from lime.lime_tabular import LimeTabularExplainer
        self.cols, self.num, self.cat = list(Xtr.columns), list(num), list(cat)
        self.levels = {c: sorted(Xtr[c].astype(str).unique()) for c in self.cat}
        self.median = {c: float(pd.to_numeric(Xtr[c], errors="coerce").median()) for c in self.num}
        self.explainer = LimeTabularExplainer(
            self._encode(Xtr), feature_names=self.cols,
            categorical_features=[self.cols.index(c) for c in self.cat],
            categorical_names={self.cols.index(c): self.levels[c] for c in self.cat},
            class_names=["Négatif", "Positif"], mode="classification",
            discretize_continuous=True, random_state=seed)

    def _encode(self, X):
        Z = pd.DataFrame(index=X.index)
        for c in self.cols:
            if c in self.cat:
                codes = {v: i for i, v in enumerate(self.levels[c])}
                Z[c] = X[c].astype(str).map(codes).fillna(0).astype(float)
            else:
                Z[c] = pd.to_numeric(X[c], errors="coerce").fillna(self.median[c]).astype(float)
        return Z.to_numpy(dtype=float)

    def _decode(self, A):
        Z = pd.DataFrame(A, columns=self.cols)
        for c in self.cat:
            idx = np.clip(np.rint(Z[c].to_numpy()).astype(int), 0, len(self.levels[c]) - 1)
            Z[c] = np.array(self.levels[c], dtype=object)[idx]
        return Z

    def explain(self, pipe, row, n_features=10, n_samples=2000):
        fn = lambda A: pipe.predict_proba(self._decode(A))
        x = self._encode(row.to_frame().T)[0]
        return self.explainer.explain_instance(x, fn, num_features=n_features,
                                               num_samples=n_samples, labels=(1,))


def _fmt(v):
    if isinstance(v, (int, np.integer)):
        return str(v)
    if isinstance(v, (float, np.floating)):
        return "n/a" if np.isnan(v) else (f"{v:.0f}" if float(v).is_integer() else f"{v:.2f}")
    return str(v)


def compare_shap_lime(shap_row, lime_w, row, sensitive=(), k=5, n=8):
    """Tableau SHAP / LIME par variable d'origine pour un dossier, et mesures d'accord."""
    lime = pd.Series(lime_w, dtype=float)
    vars_ = list(dict.fromkeys(list(shap_row.abs().sort_values(ascending=False).head(n).index)
                               + list(lime.abs().sort_values(ascending=False).head(n).index)))
    tab = pd.DataFrame({"Variable": vars_,
                        "Valeur du dossier": [_fmt(row[v]) for v in vars_],
                        "SHAP": [float(shap_row.get(v, 0.0)) for v in vars_],
                        "LIME": [float(lime.get(v, 0.0)) for v in vars_]})
    both = (tab["SHAP"] != 0) & (tab["LIME"] != 0)
    same = np.sign(tab["SHAP"]) == np.sign(tab["LIME"])
    tab["Même sens"] = np.where(~both, "—", np.where(same, "oui", "non"))
    tab["Sensible"] = np.where(tab["Variable"].isin(sensitive), "oui", "")
    top_s = set(shap_row.abs().nlargest(k).index)
    top_l = set(lime.abs().nlargest(k).index)
    return dict(table=tab, k=k, n_common=len(top_s & top_l),
                n_agree=int((both & same).sum()), n_both=int(both.sum()))


def summary_sentence(proba, thr, contrib, row, k_up=2, k_down=1):
    """Phrase de synthèse en français à partir des contributions par variable d'origine."""
    flag = proba >= thr
    label = "À investiguer" if flag else "Traitement standard"
    f = lambda v: f"{v} = {_fmt(row[v])}"
    up = contrib[contrib > 0].sort_values(ascending=False).head(k_up).index
    down = contrib[contrib < 0].sort_values().head(k_down if len(up) else k_up).index
    txt = (f"Ce dossier est classé « {label} » : la probabilité de fraude estimée est de {proba * 100:.0f} % "
           f"pour un seuil de {thr * 100:.0f} %.")
    if len(up) and len(down):
        txt += f" Elle est tirée vers le haut surtout par {' et '.join(f(v) for v in up)}, et vers le bas par {' et '.join(f(v) for v in down)}."
    elif len(up):
        txt += f" Elle est tirée vers le haut surtout par {' et '.join(f(v) for v in up)}."
    elif len(down):
        txt += f" Elle est tirée vers le bas surtout par {' et '.join(f(v) for v in down)}."
    return txt + " Ces éléments sont des associations apprises par le modèle, pas des causes."


# ======================= API d'interprétation (étape 3) =======================

def explain_instance(bundle, name, variables, thr=0.5, top_k=8, sensitive=()):
    """Explication SHAP d'un dossier (un dict de variables), regroupée par variable d'origine."""
    if name not in SHAP_MODELS:
        raise ValueError(f"SHAP non disponible pour {name}. Modèles expliqués : {', '.join(SHAP_MODELS)}")
    pipe = bundle["models"][name]
    row = pd.DataFrame([variables])[bundle["features"]]
    proba = float(pipe.predict_proba(row)[0, 1])
    S = shap_values(name, pipe, bundle["background"], row, bundle["cat"])
    contrib = group_by_origin(S["vals"], S["origin"]).iloc[0]
    order = contrib.abs().sort_values(ascending=False).index
    r = row.iloc[0]
    top = [dict(variable=v, valeur=_fmt(r[v]), contribution=round(float(contrib[v]), 5),
                effet="augmente le risque" if contrib[v] > 0 else "diminue le risque",
                sensible=v in sensitive) for v in order[:top_k]]
    return dict(modele=name, probabilite_fraude=round(proba, 4), seuil=thr,
                decision="À investiguer" if proba >= thr else "Traitement standard",
                unite=scale_label(name), valeur_de_base=round(float(S["base"][0]), 5),
                somme_des_contributions=round(float(contrib.sum()), 5),
                contributions=top, phrase=summary_sentence(proba, thr, contrib, r))
    
