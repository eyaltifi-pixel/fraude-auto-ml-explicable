"""Onglet « 5 · Explicabilité » : explications globales (étape 1)."""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import matplotlib.pyplot as plt
from core import SENSITIVE, make_sig
import xai_core as xc

COLORS = {"Variable sensible": "#E07B39", "Autre variable": "#1F3A5F"}


def cached(key, fn):
    store = st.session_state.setdefault("xai_cache", {})
    if key not in store:
        store[key] = fn()
    return store[key]


def hbar(df, value_col, label_col, title, origin_col=None, n=15):
    """Barres horizontales ; les variables sensibles sont en orange."""
    d = df.head(n).copy()
    d["Variable"] = d[label_col]
    org = d[origin_col] if origin_col else d[label_col]
    d["Type"] = np.where(org.isin(SENSITIVE), "Variable sensible", "Autre variable")
    fig = px.bar(d, x=value_col, y="Variable", orientation="h", color="Type",
                 color_discrete_map=COLORS, title=title)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=max(300, 26 * len(d) + 120), legend_title_text="")
    return fig


def render_global(P, R, sig):
    import shap
    fitted, Xtr, Xte, yte, cat = R["fitted"], P["X_train"], P["X_test"], P["y_test"], P["cat"]
    models = list(fitted)
    model = st.selectbox("Modèle à expliquer", models,
                         index=models.index("XGBoost") if "XGBoost" in models else 0, key="xai_model")

    # ---------- 1. importance propre au modèle ----------
    st.markdown("#### 1. Importance propre au modèle")
    df, kind = xc.intrinsic_importance(model, fitted[model], cat)
    if df is None:
        st.info(kind)
    else:
        st.caption(kind)
        fig = hbar(df.reset_index(names="nom"), "valeur", "nom", f"Importance propre au modèle : {model}",
                   origin_col="origine")
        st.plotly_chart(fig, width="stretch")

    # ---------- 2. importance par permutation ----------
    st.markdown("#### 2. Importance par permutation (comparable entre modèles)")
    st.caption("On mélange une variable du jeu de test et on mesure la baisse de PR-AUC : plus la baisse est forte, "
               "plus le modèle s'appuie sur cette variable. Même échelle pour tous les modèles.")
    pkey = ("perm", sig, tuple(fitted))
    if pkey not in st.session_state.get("xai_cache", {}):
        if st.button("Calculer l'importance par permutation (tous les modèles)", key="xai_perm_btn"):
            with st.spinner("Permutations en cours…"):
                cached(pkey, lambda: xc.permutation_table(fitted, Xte, yte))
            st.rerun()
    else:
        T = st.session_state["xai_cache"][pkey]
        top = T.mean(axis=1).sort_values(ascending=False).head(10).index
        long = T.loc[top].reset_index(names="Variable").melt(id_vars="Variable", var_name="Modèle", value_name="Baisse de PR-AUC")
        fig = px.bar(long, x="Baisse de PR-AUC", y="Variable", color="Modèle", orientation="h", barmode="group",
                     title="10 variables les plus importantes (moyenne des modèles)")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=520)
        st.plotly_chart(fig, width="stretch")
        sens = [c for c in SENSITIVE if c in T.index]
        if sens:
            st.markdown("**Variables sensibles : importance par permutation**")
            st.dataframe(T.loc[sens].round(4), width="stretch")
            st.caption("Une valeur proche de 0 (ou négative) signifie que le modèle ne s'en sert pas. "
                       "Sous 0,01, on est dans le bruit pour un jeu de test de cette taille.")

    # ---------- 3. SHAP ----------
    st.markdown("#### 3. Explications SHAP")
    if model not in xc.SHAP_MODELS:
        st.info("SHAP n'est pas calculé pour KNN (trop coûteux, et modèle peu performant ici). Choisis un autre modèle.")
        return
    with st.spinner("Calcul des valeurs SHAP sur le jeu de test…"):
        S = cached(("shap", model, sig), lambda: xc.shap_values(model, fitted[model], Xtr, Xte, cat))
    st.caption(f"Unité des contributions SHAP pour ce modèle : **{xc.scale_label(model)}**. "
               "Une valeur positive pousse vers la classe positive (fraude).")

    st.markdown("**Graphique SHAP summary (beeswarm)**")
    expl = shap.Explanation(values=S["vals"], base_values=S["base"], data=S["T"].to_numpy(), feature_names=S["names"])
    plt.close("all")
    shap.plots.beeswarm(expl, max_display=15, show=False)
    st.pyplot(plt.gcf(), clear_figure=True)
    st.caption("Chaque point est un dossier du test. Position horizontale : effet sur la prédiction. "
               "Couleur : valeur de la variable (rouge = élevée ; pour une catégorie one-hot, rouge = présente).")

    G = xc.group_by_origin(S["vals"], S["origin"])
    imp = G.abs().mean().sort_values(ascending=False)
    imp_df = pd.DataFrame({"Variable": imp.index, "Contribution SHAP moyenne (valeur absolue)": imp.to_numpy()})
    st.plotly_chart(hbar(imp_df, "Contribution SHAP moyenne (valeur absolue)", "Variable",
                         "Importance SHAP par variable d'origine (catégories regroupées)", origin_col="Variable"),
                    width="stretch")
    sens = [c for c in SENSITIVE if c in imp.index]
    if sens:
        share = float(imp[sens].sum() / imp.sum())
        st.metric("Part des variables sensibles dans l'importance SHAP totale", f"{share:.1%}")
        st.caption("Variables sensibles présentes : " + ", ".join(f"{c} ({imp[c] / imp.sum():.1%})" for c in sens))

    st.markdown("**Graphiques de dépendance SHAP (4 variables principales)**")
    cols = st.columns(2)
    for i, v in enumerate(imp.head(4).index):
        d = pd.DataFrame({v: Xte[v].to_numpy(), "Contribution SHAP": G[v].to_numpy(),
                          "Classe réelle": np.where(yte.to_numpy() == 1, "Positif", "Négatif")})
        if v in P["num"]:
            fig = px.scatter(d, x=v, y="Contribution SHAP", color="Classe réelle", opacity=0.7)
        else:
            fig = px.box(d, x=v, y="Contribution SHAP", points="all", color="Classe réelle")
            fig.update_xaxes(categoryorder="median descending")
        fig.add_hline(y=0, line_dash="dash")
        fig.update_layout(title=f"Dépendance SHAP : {v}", height=380)
        cols[i % 2].plotly_chart(fig, width="stretch")
    st.caption("Variable numérique : chaque point montre comment la contribution change avec la valeur. "
               "Variable catégorielle : distribution de la contribution selon la modalité. "
               "Ce sont des associations apprises par le modèle, pas des causes.")


def render():
    st.subheader("Explicabilité (XAI)")
    if "models" not in st.session_state or "prep" not in st.session_state:
        st.info("Entraîne d'abord les modèles dans l'onglet 3 · Modélisation.")
        return
    P, R = st.session_state["prep"], st.session_state["models"]
    if R.get("sig") != make_sig(P):
        st.warning("Les réglages de préparation ont changé depuis l'entraînement : "
                   "relance « Entraîner et comparer » dans l'onglet 3.")
        return
    sig = (hash(make_sig(P)), hash(str(R["params"]) + R["table"].to_csv()))
    st.markdown("### Explications globales : le modèle dans son ensemble")
    render_global(P, R, sig)
    render_local(P, R, sig, st.session_state.get("xai_model"))


# ======================= Explications locales (étape 2) =======================

def render_local(P, R, sig, model):
    import shap
    fitted, Xtr, Xte, yte = R["fitted"], P["X_train"], P["X_test"], P["y_test"]
    num, cat = P["num"], P["cat"]
    if model not in fitted:
        return
    pipe = fitted[model]
    st.markdown("---")
    st.markdown("### Explications locales : un dossier précis")

    # ---------- choix du dossier et du seuil ----------
    a, b, c = st.columns([1, 2, 2])
    kind = a.selectbox("Dossiers", ["Tous", "Fraudes réelles", "Non-fraudes"], key="xai_kind")
    mask = np.ones(len(Xte), dtype=bool)
    if kind == "Fraudes réelles":
        mask = yte.to_numpy() == 1
    elif kind == "Non-fraudes":
        mask = yte.to_numpy() == 0
    idx = b.selectbox("Dossier du jeu de test (numéro de ligne du jeu de données)", list(Xte.index[mask]),
                      key=f"xai_idx_{kind}")
    thr = c.slider("Seuil de décision", 0.05, 0.95, 0.50, 0.05, key="xai_thr")
    pos = Xte.index.get_loc(idx)
    row = Xte.loc[idx]

    proba = float(pipe.predict_proba(Xte.loc[[idx]])[0, 1])
    m1, m2, m3 = st.columns(3)
    m1.metric("Probabilité prédite", f"{proba:.1%}")
    m2.metric("Décision au seuil choisi", "À investiguer" if proba >= thr else "Traitement standard")
    m3.metric("Vérité terrain", "Positif (fraude)" if yte.loc[idx] == 1 else "Négatif")
    with st.expander("Valeurs des variables de ce dossier"):
        st.dataframe(row.astype(str).rename("valeur").to_frame(), width="stretch")

    # ---------- SHAP waterfall ----------
    st.markdown("#### Graphique SHAP waterfall")
    contrib = None
    if model in xc.SHAP_MODELS:
        S = cached(("shap", model, sig), lambda: xc.shap_values(model, pipe, Xtr, Xte, cat))
        contrib = xc.group_by_origin(S["vals"], S["origin"]).iloc[pos]
        D = xc.display_data(S, Xte, num)
        expl = shap.Explanation(values=S["vals"][pos], base_values=float(S["base"][pos]),
                                data=D[pos], feature_names=S["names"])
        plt.close("all")
        shap.plots.waterfall(expl, max_display=12, show=False)
        st.pyplot(plt.gcf(), clear_figure=True)
        st.caption(f"Lecture : on part de la valeur moyenne E[f(X)] et chaque barre ajoute ou retire la contribution "
                   f"d'une variable jusqu'à f(x). Unité : {xc.scale_label(model)}. "
                   "Les variables catégorielles sont en one-hot (« variable_modalité = 1 » : la modalité est présente).")
    else:
        st.info("SHAP n'est pas calculé pour KNN. L'explication LIME ci-dessous reste disponible.")

    # ---------- LIME ----------
    st.markdown("#### Explication LIME du même dossier")
    lime_obj = cached(("limeobj", sig), lambda: xc.LimeRaw(Xtr, num, cat))
    with st.spinner("Calcul de l'explication LIME…"):
        exp = cached(("lime", model, sig, idx), lambda: lime_obj.explain(pipe, row))
    items = exp.as_list(label=1)
    lime_w = {lime_obj.cols[j]: w for j, w in exp.as_map()[1]}
    ld = pd.DataFrame(items, columns=["Condition", "Poids"])
    ld["Effet"] = np.where(ld["Poids"] > 0, "Vers la classe positive", "Vers la classe négative")
    fig = px.bar(ld, x="Poids", y="Condition", orientation="h", color="Effet",
                 color_discrete_map={"Vers la classe positive": "#E07B39", "Vers la classe négative": "#1F3A5F"},
                 title="Poids LIME (effet local sur la probabilité de la classe positive)")
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=420, legend_title_text="")
    st.plotly_chart(fig, width="stretch")
    try:
        sc = exp.score.get(1) if isinstance(exp.score, dict) else exp.score
        st.caption(f"Fidélité locale du modèle LIME (R²) : {float(sc):.2f}. "
                   "Proche de 1 : l'approximation locale est fiable ; proche de 0 : à prendre avec prudence.")
    except Exception:
        pass

    # ---------- comparaison SHAP / LIME ----------
    if contrib is not None:
        st.markdown("#### Comparaison SHAP / LIME")
        cmp = xc.compare_shap_lime(contrib, lime_w, row, sensitive=SENSITIVE)
        k1, k2 = st.columns(2)
        k1.metric(f"Variables communes dans le top {cmp['k']}", f"{cmp['n_common']} / {cmp['k']}")
        k2.metric("Même sens (SHAP et LIME)", f"{cmp['n_agree']} / {cmp['n_both']}")
        tab = cmp["table"]
        st.dataframe(tab.round(3), hide_index=True, width="stretch")
        norm = tab.assign(SHAP=tab["SHAP"] / max(tab["SHAP"].abs().max(), 1e-12),
                          LIME=tab["LIME"] / max(tab["LIME"].abs().max(), 1e-12))
        long = norm.melt(id_vars="Variable", value_vars=["SHAP", "LIME"], var_name="Méthode",
                         value_name="Contribution normalisée")
        fig = px.bar(long, x="Contribution normalisée", y="Variable", color="Méthode", orientation="h",
                     barmode="group", title="SHAP et LIME par variable (chacun divisé par son maximum en valeur absolue)")
        fig.update_yaxes(autorange="reversed")
        fig.update_layout(height=420)
        st.plotly_chart(fig, width="stretch")
        st.caption("SHAP et LIME n'ont pas la même unité : on compare surtout le **sens** et le **rang** des variables, "
                   "pas les valeurs brutes. Un « — » signifie que la variable est hors du top 10 de LIME. "
                   "Un désaccord de signe est fréquent pour les variables à faible contribution : "
                   "LIME est une approximation locale par échantillonnage, SHAP s'appuie sur le modèle lui-même.")

    # ---------- phrase de synthèse ----------
    st.markdown("#### Phrase de synthèse")
    src = contrib if contrib is not None else pd.Series(lime_w, dtype=float)
    st.info(xc.summary_sentence(proba, thr, src, row))
