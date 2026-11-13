# -*- coding: utf-8 -*-
"""Pipeline completo del articulo. Genera TODAS las tablas, figuras y cifras
que aparecen en el texto.

    python pipeline.py

Salidas:
    04_resultados/paper20260905/tablas.xlsx      una hoja por tabla del articulo
    04_resultados/paper20260905/valores_paper.json  cada cifra citada en el texto
    05_figuras/paper20260905/figNN_*.png / .pdf
"""
import json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import (spearmanr, pearsonr, shapiro, levene, kruskal,
                         mannwhitneyu, rankdata)
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (silhouette_score, adjusted_rand_score, roc_auc_score,
                             r2_score, mean_absolute_error, roc_curve)
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (StratifiedKFold, KFold, cross_val_predict,
                                     RepeatedStratifiedKFold)
import statsmodels.api as sm

import config as C
from hoeffding import hoeffding_d

warnings.filterwarnings("ignore")
rng = np.random.default_rng(C.SEMILLA)
V = {}          # cifras citadas en el articulo
TABLAS = {}     # hoja -> DataFrame


def fig(nombre):
    for ext in ("png", "pdf"):
        plt.savefig(f"{C.FIG}/{nombre}.{ext}", dpi=160, bbox_inches="tight")
    plt.close()
    print(f"    figura {nombre}")


def norm(df, cols):
    """Transformacion cuantil a normal + estandarizacion (robusta a los extremos)."""
    Q = QuantileTransformer(output_distribution="normal", n_quantiles=len(df),
                            random_state=C.SEMILLA)
    return StandardScaler().fit_transform(Q.fit_transform(df[cols]))


# =============================================================================
print("\n[1] Datos y auditoria")
# =============================================================================
bruto = pd.read_excel(C.ARCHIVO_DATOS).rename(columns={"succes_ratio": "success_ratio"})
V["n_filas_archivo"] = len(bruto)
dupes = bruto.file_name.duplicated(keep=False)
V["n_nombres_duplicados"] = int(bruto.loc[dupes, "file_name"].nunique())
V["n_filas_duplicadas"] = int(dupes.sum())

num = bruto.select_dtypes(include=[np.number]).columns
df = bruto.groupby("file_name", as_index=False).agg({**{c: "mean" for c in num},
                                                     "source": "first"})
V["n_piezas"] = len(df)

# --- auditoria de las variables inservibles (C31)
aud = []
for c in C.CORRUPTAS:
    aud.append({"variable": c, "minimo": bruto[c].min(), "maximo": bruto[c].max(),
                "valores_unicos": bruto[c].nunique(),
                "rango_esperado": "0/1" if c == "is_surface_model" else "conteo (10-10^3)"})
TABLAS["A1_variables_corruptas"] = pd.DataFrame(aud)
V["is_surface_model_max"] = float(bruto.is_surface_model.max())
V["is_surface_model_binarias"] = int((bruto.is_surface_model <= 1).sum())

# --- mass = volume x densidad por defecto (C33)
dens = df.mass / df.volume
V["densidad_mediana"] = float(dens.median())
V["n_densidad_1000"] = int((dens.round(-1) == 1000).sum())
V["pct_densidad_1000"] = round(100 * (dens.round(-1) == 1000).mean(), 1)
V["rho_mass_volume"] = round(float(spearmanr(df.mass, df.volume).statistic), 3)

# --- success_ratio (C41)
V["n_ceros"] = int((df.success_ratio == 0).sum())
V["pct_ceros"] = round(100 * (df.success_ratio == 0).mean(), 1)
pos = df.loc[df.success_ratio > 0, "success_ratio"]
V["sr_pos_mediana"] = round(float(pos.median()), 3)
V["sr_pos_media"] = round(float(pos.mean()), 3)
V["sr_pos_max"] = round(float(pos.max()), 3)
V["n_positivas"] = int((df.success_ratio > 0).sum())

V["fuentes"] = df.source.nunique()
TABLAS["A2_fuentes"] = (df.groupby("source")
    .agg(n=("file_name", "size"),
         ceros=("success_ratio", lambda s: int((s == 0).sum())),
         pct_cero=("success_ratio", lambda s: round(100 * (s == 0).mean(), 1)),
         sr_mediana=("success_ratio", "median"))
    .sort_values("n", ascending=False).reset_index())
print(f"    {V['n_filas_archivo']} filas -> {V['n_piezas']} piezas; "
      f"{V['pct_ceros']}% con success_ratio = 0")

# --- Tabla 1 del articulo: definicion de variables
DEF = {
 "cyclomatic_complexity": ("grafo", "Complejidad ciclomatica del grafo de dependencias padre/hijo: numero de caminos linealmente independientes, E - N + 2."),
 "modularity":            ("grafo", "Modularidad de Newman del grafo de dependencias: grado en que el arbol se organiza en subconjuntos de operaciones acopladas entre si."),
 "graph_density":         ("grafo", "Densidad del grafo: aristas observadas sobre aristas posibles. Valores bajos indican arboles alargados y poco ramificados."),
 "entropy_wang":          ("grafo", "Entropia de Wang del grafo de dependencias: mide la heterogeneidad de la distribucion de grados de los nodos."),
 "entropy_shannon":       ("grafo", "Entropia de Shannon sobre la distribucion de tipos de operacion del arbol parametrico."),
 "entropy_infoshannon":   ("grafo", "Entropia de Shannon normalizada por el maximo teorico (log del numero de tipos de operacion presentes)."),
 "volume_area_ratio":     ("geometrica", "Cociente entre el volumen de la pieza y su area superficial; describe la compacidad de la forma."),
 "volume_ratio":          ("geometrica", "1 - (volumen de la pieza / volumen de su caja envolvente)."),
 "sphere_ratio":          ("geometrica", "1 - (volumen de la pieza / volumen de la esfera envolvente)."),
 "size":                  ("archivo", "Tamano del fichero nativo .sldprt en kilobytes; recoge de forma agregada el volumen de informacion almacenada."),
 "success_ratio":         ("calidad", "Variable objetivo de calidad. Proporcion de modificaciones dimensionales exhaustivas que el modelo regenera sin error, normalizada al incremento maximo ensayado; sigue el procedimiento del Ratio for Exhaustive Modification."),
}
TABLAS["T1_variables"] = pd.DataFrame(
    [{"variable": k, "familia": f, "definicion": d} for k, (f, d) in DEF.items()])

# --- Tabla 2: descriptivos
d2 = df[C.VARS + [C.OBJETIVO]].describe().T[["mean", "50%", "std", "min", "max"]]
d2.columns = ["media", "mediana", "desv", "minimo", "maximo"]
d2.insert(0, "familia", [C.FAMILIA.get(i, "calidad") for i in d2.index])
TABLAS["T2_descriptivos"] = d2.reset_index().rename(columns={"index": "variable"})

# --- Figura 1: distribuciones
f, ax = plt.subplots(2, 5, figsize=(17, 7))
for a, v in zip(ax.ravel(), C.VARS):
    a.violinplot(df[v], showmedians=True)
    a.set_title(v, fontsize=9); a.set_xticks([]); a.tick_params(labelsize=7)
f.suptitle(f"Figura 1. Distribucion de las diez metricas del estudio (n = {len(df)})")
f.tight_layout(); fig("fig01_distribuciones")


# =============================================================================
print("\n[2] Estructura de dependencia y seleccion de variables")
# =============================================================================
CAND = C.GRAFO + C.GEOMETRIA + C.ARCHIVO + ["kolmogorov_cl", "cube_ratio",
                                            "volume", "surface", "mass", "time"]
filas = []
for i, a in enumerate(CAND):
    for b in CAND[i + 1:]:
        filas.append({"var_x": a, "var_y": b,
                      "pearson": pearsonr(df[a], df[b])[0],
                      "spearman": spearmanr(df[a], df[b]).statistic,
                      "hoeffding_D": hoeffding_d(df[a], df[b])})
dep_all = pd.DataFrame(filas)
red = []
for v, sust in C.REDUNDANTES.items():
    r = dep_all[((dep_all.var_x == v) & (dep_all.var_y == sust)) |
                ((dep_all.var_y == v) & (dep_all.var_x == sust))].iloc[0]
    red.append({"descartada": v, "redundante_con": sust,
                "spearman": r.spearman, "hoeffding_D": r.hoeffding_D})
TABLAS["T3_redundancia"] = pd.DataFrame(red)
_vam = dep_all[((dep_all.var_x == "volume_area_ratio") & (dep_all.var_y == "mass")) |
               ((dep_all.var_y == "volume_area_ratio") & (dep_all.var_x == "mass"))].iloc[0]
V["pearson_varea_mass"] = round(float(_vam.pearson), 3)
V["spearman_varea_mass"] = round(float(_vam.spearman), 3)
V["rho_time_wang"] = round(float(abs(dep_all[((dep_all.var_x == "time") & (dep_all.var_y == "entropy_wang")) |
                                                ((dep_all.var_y == "time") & (dep_all.var_x == "entropy_wang"))].spearman.iloc[0])), 3)
V["rho_kolmogorov_wang"] = round(float(abs(TABLAS["T3_redundancia"].iloc[0].spearman)), 3)
V["rho_cube_sphere"] = round(float(abs(TABLAS["T3_redundancia"].iloc[1].spearman)), 3)

# --- dependencia entre las 10 finales + objetivo (C34)
VV = C.VARS + [C.OBJETIVO]
n = len(VV)
S = pd.DataFrame(np.eye(n), index=VV, columns=VV)
D = pd.DataFrame(np.eye(n), index=VV, columns=VV)
pares = []
for i in range(n):
    for j in range(i + 1, n):
        rho = spearmanr(df[VV[i]], df[VV[j]]).statistic
        d_ = hoeffding_d(df[VV[i]], df[VV[j]])
        S.iloc[i, j] = S.iloc[j, i] = rho
        D.iloc[i, j] = D.iloc[j, i] = d_
        pares.append({"var_x": VV[i], "var_y": VV[j], "spearman": rho, "hoeffding_D": d_,
                      "pearson": pearsonr(df[VV[i]], df[VV[j]])[0]})
PAR = pd.DataFrame(pares)
TABLAS["T4_dependencias"] = PAR.sort_values("hoeffding_D", ascending=False)
V["n_pares"] = len(PAR)
V["n_pares_no_monotonos"] = int(((PAR.hoeffding_D > .03) & (PAR.spearman.abs() < .30)).sum())
V["corr_D_rho"] = round(float(PAR.hoeffding_D.corr(PAR.spearman.abs())), 3)
V["rho_wang_ciclomatica"] = round(float(PAR[(PAR.var_x == "cyclomatic_complexity") &
                                            (PAR.var_y == "entropy_wang")].spearman.iloc[0]), 3)
print(f"    pares con dependencia no monotona: {V['n_pares_no_monotonos']} de {V['n_pares']}"
      f"   corr(D,|rho|) = {V['corr_D_rho']}")

f, ax = plt.subplots(1, 2, figsize=(13.5, 5.6))
i0 = ax[0].imshow(S.values, cmap="RdBu_r", vmin=-1, vmax=1)
ax[0].set_title("(a) Spearman $\\rho$")
mx = float(np.nanmax(D.values[~np.eye(n, dtype=bool)]))
i1 = ax[1].imshow(D.values, cmap="viridis", vmin=0, vmax=mx)
ax[1].set_title("(b) Hoeffding's D")
for a, im in zip(ax, (i0, i1)):
    a.set_xticks(range(n)); a.set_xticklabels(VV, rotation=90, fontsize=7)
    a.set_yticks(range(n)); a.set_yticklabels(VV, fontsize=7)
    f.colorbar(im, ax=a, fraction=.046)
f.suptitle("Figura 2. Dependencia monotona (Spearman) y dependencia general (Hoeffding's D)")
f.tight_layout(); fig("fig02_spearman_hoeffding")


# =============================================================================
print("\n[3] Estructura del espacio de metricas: ¿grupos o continuo?")
# =============================================================================
X = norm(df, C.VARS)
pca_full = PCA(random_state=C.SEMILLA).fit(X)
var_acum = np.cumsum(pca_full.explained_variance_ratio_)
k90 = int(np.searchsorted(var_acum, .90) + 1)
Z = PCA(n_components=k90, random_state=C.SEMILLA).fit_transform(X)
V["pca_pc1"] = round(float(pca_full.explained_variance_ratio_[0] * 100), 1)
V["pca_pc2"] = round(float(pca_full.explained_variance_ratio_[1] * 100), 1)
V["pca_pc12"] = round(V["pca_pc1"] + V["pca_pc2"], 1)
V["pca_k90"] = k90
TABLAS["T5_pca_cargas"] = pd.DataFrame(
    PCA(n_components=3, random_state=C.SEMILLA).fit(X).components_.T,
    index=C.VARS, columns=["PC1", "PC2", "PC3"]).reset_index().rename(columns={"index": "variable"})

def W(Zm, k):
    km = KMeans(k, n_init=25, random_state=C.SEMILLA).fit(Zm)
    return sum(((Zm[km.labels_ == i] - Zm[km.labels_ == i].mean(0)) ** 2).sum() for i in range(k))

ks, B = list(range(1, 7)), 30
mn, mx_ = Z.min(0), Z.max(0)
gap, sk, sil = [], [], {}
for k in ks:
    refs = [np.log(W(rng.uniform(mn, mx_, Z.shape), k)) for _ in range(B)]
    gap.append(np.mean(refs) - np.log(W(Z, k))); sk.append(np.std(refs) * np.sqrt(1 + 1 / B))
    if k > 1:
        sil[k] = silhouette_score(Z, KMeans(k, n_init=50, random_state=C.SEMILLA).fit(Z).labels_)
gap, sk = np.array(gap), np.array(sk)
k_opt = next((k for i, k in enumerate(ks) if i + 1 < len(ks) and gap[i] >= gap[i + 1] - sk[i + 1]), None)
TABLAS["T6_estructura"] = pd.DataFrame({"k": ks, "gap": gap, "s_k": sk,
                                        "silueta": [np.nan] + [sil[k] for k in ks[1:]]})
V["gap_k_optimo"] = int(k_opt)
V["gap_k1"] = round(float(gap[0]), 3); V["gap_k2"] = round(float(gap[1]), 3)
V["silueta_k2"] = round(float(sil[2]), 3)
nulos = [silhouette_score(u, KMeans(2, n_init=20, random_state=C.SEMILLA).fit(u).labels_)
         for u in (rng.uniform(mn, mx_, Z.shape) for _ in range(20))]
V["silueta_nula"] = round(float(np.mean(nulos)), 3)
V["silueta_nula_sd"] = round(float(np.std(nulos)), 3)

km2 = KMeans(2, n_init=50, random_state=C.SEMILLA).fit(Z)
lab = km2.labels_
if df.cyclomatic_complexity[lab == 0].median() > df.cyclomatic_complexity[lab == 1].median():
    lab = 1 - lab
df["etiqueta_binaria"] = lab
ward = AgglomerativeClustering(n_clusters=2, linkage="ward").fit(Z)
V["ari_kmeans_ward"] = round(float(adjusted_rand_score(lab, ward.labels_)), 3)
print(f"    gap statistic -> k = {k_opt}; silueta k=2 = {V['silueta_k2']} "
      f"(nulo uniforme {V['silueta_nula']})")

f, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].plot(range(1, len(var_acum) + 1), var_acum * 100, "o-")
ax[0].axhline(90, ls="--", c="gray"); ax[0].set_xlabel("componentes")
ax[0].set_ylabel("% varianza acumulada"); ax[0].set_title("(a) Varianza explicada")
ax[1].errorbar(ks, gap, yerr=sk, fmt="o-", capsize=3)
ax[1].set_xlabel("numero de grupos k"); ax[1].set_ylabel("gap")
ax[1].set_title(f"(b) Gap statistic: optimo en k = {k_opt}")
ax[2].plot(list(sil), list(sil.values()), "o-", label="observado")
ax[2].axhline(V["silueta_nula"], ls="--", c="crimson", label="datos uniformes sin estructura")
ax[2].set_xlabel("numero de grupos k"); ax[2].set_ylabel("silueta"); ax[2].legend(fontsize=8)
ax[2].set_title("(c) Silueta")
f.suptitle("Figura 3. El espacio de metricas no contiene grupos separables")
f.tight_layout(); fig("fig03_estructura")


# =============================================================================
print("\n[4] Indice continuo de complejidad y validacion externa")
# =============================================================================
pc1 = PCA(n_components=1, random_state=C.SEMILLA).fit(X)
s = pc1.transform(X)[:, 0]
signo = np.sign(spearmanr(s, df.cyclomatic_complexity).statistic)
df["ICE"] = s * signo                       # Indice de Complejidad Estructural
TABLAS["T7_ice_cargas"] = pd.DataFrame({"variable": C.VARS,
                                        "carga_PC1": pc1.components_[0] * signo}
                                       ).sort_values("carga_PC1", key=abs, ascending=False)
for _v, _c in TABLAS["T7_ice_cargas"].set_index("variable").carga_PC1.items():
    V[f"carga_PC1_{_v}"] = round(float(_c), 3)
df["ICE_promedio"] = X.mean(axis=1) * signo   # el indice del borrador anterior

smap = pd.read_excel(C.ARCHIVO_SMAP, sheet_name=C.HOJA_SMAP)
df["id"] = df.file_name.str.extract(r"^(\d+)").astype(float)
M = df.merge(smap[["id", "s_map"]], on="id", how="inner")
V["n_smap"] = len(M)

# tasas de exito de las dos mediciones del unico fichero duplicado (seccion 3.2)
_dup = pd.read_excel(C.ARCHIVO_DATOS)
_dup = _dup[_dup.duplicated(subset="file_name", keep=False)].sort_values("succes_ratio", ascending=False)
if len(_dup) == 2:
    V["dup_sr_alto"] = round(float(_dup.succes_ratio.iloc[0]), 3)
    V["dup_sr_bajo"] = round(float(_dup.succes_ratio.iloc[1]), 3)

def boot_rho(a, b, n=2000):
    v = [spearmanr(a.iloc[i], b.iloc[i]).statistic
         for i in (rng.integers(0, len(a), len(a)) for _ in range(n))]
    return np.percentile(v, [2.5, 97.5])

# Indice con pesos ajustados por regresion de cresta contra S_MAP, evaluado
# fuera de muestra (5 particiones) para que el ajuste no infle su desempeno.
from sklearn.linear_model import RidgeCV
Xg_smap = norm(df, C.GRAFO)[df.id.isin(smap.id).values]
M["ICE_ridge_cv"] = cross_val_predict(RidgeCV(alphas=np.logspace(-3, 3, 25)), Xg_smap,
                                      M.s_map.values, cv=KFold(5, shuffle=True, random_state=C.SEMILLA))
_ridge = RidgeCV(alphas=np.logspace(-3, 3, 25)).fit(Xg_smap, M.s_map.values)
TABLAS["T8b_pesos_ridge"] = (pd.Series(_ridge.coef_, index=C.GRAFO)
                             .sort_values(key=abs, ascending=False)
                             .rename("peso").reset_index().rename(columns={"index": "variable"}))

# Indice restringido a las seis metricas de grafo: la cuarta construccion que
# anuncia la seccion 3.5 y que la revision pedia comparar explicitamente.
_Xg = norm(df, C.GRAFO)
_pc1g = PCA(n_components=1, random_state=C.SEMILLA).fit(_Xg)
_sg = _pc1g.transform(_Xg)[:, 0]
_sg = _sg * np.sign(spearmanr(_sg, df.cyclomatic_complexity).statistic)
df["ICE_grafo"] = _sg
M["ICE_grafo"] = df.loc[df.id.isin(smap.id).values, "ICE_grafo"].values
V["pca_pc1_grafo"] = round(float(_pc1g.explained_variance_ratio_[0] * 100), 1)

comp = []
for nom, col in [("Etiqueta binaria (k-means)", "etiqueta_binaria"),
                 ("Promedio de variables estandarizadas", "ICE_promedio"),
                 ("Pesos ajustados por regresion de cresta (fuera de muestra)", "ICE_ridge_cv"),
                 ("PC1 de las seis metricas de grafo", "ICE_grafo"),
                 ("ICE (primera componente principal)", "ICE")]:
    r = spearmanr(M[col], M.s_map)
    lo, hi = boot_rho(M[col], M.s_map)
    comp.append({"predictor de S_MAP": nom, "spearman": r.statistic, "IC95_inf": lo,
                 "IC95_sup": hi, "varianza_%": 100 * r.statistic ** 2, "p": r.pvalue})
TABLAS["T8_validacion_smap"] = pd.DataFrame(comp)
V["rho_ice_smap"] = round(float(spearmanr(M.ICE, M.s_map).statistic), 3)
V["rho_binaria_smap"] = round(float(spearmanr(M.etiqueta_binaria, M.s_map).statistic), 3)
V["rho_promedio_smap"] = round(float(spearmanr(M.ICE_promedio, M.s_map).statistic), 3)
V["rho_ridge_smap"] = round(float(spearmanr(M.ICE_ridge_cv, M.s_map).statistic), 3)
V["rho_grafo_smap"] = round(float(spearmanr(M.ICE_grafo, M.s_map).statistic), 3)
V["var_grafo"] = round(100 * V["rho_grafo_smap"] ** 2, 1)
V["var_ridge"] = round(100 * V["rho_ridge_smap"] ** 2, 1)
V["var_ice"] = round(100 * V["rho_ice_smap"] ** 2, 1)
V["var_binaria"] = round(100 * V["rho_binaria_smap"] ** 2, 1)
V["ganancia_var"] = round(V["var_ice"] - V["var_binaria"], 1)
ic = boot_rho(M.ICE, M.s_map)
V["ic_ice_inf"], V["ic_ice_sup"] = round(float(ic[0]), 3), round(float(ic[1]), 3)
u = mannwhitneyu(M.s_map[M.etiqueta_binaria == 1], M.s_map[M.etiqueta_binaria == 0])
V["auc_binaria"] = round(float(u.statistic / ((M.etiqueta_binaria == 1).sum() *
                                              (M.etiqueta_binaria == 0).sum())), 3)

ind = []
for v in C.VARS:
    r = spearmanr(M[v], M.s_map)
    ind.append({"variable": v, "familia": C.FAMILIA[v], "spearman": r.statistic,
                "p": r.pvalue, "hoeffding_D": hoeffding_d(M[v], M.s_map)})
IND = pd.DataFrame(ind).sort_values("spearman", key=abs, ascending=False)
TABLAS["T9_variables_vs_smap"] = IND
V["mejor_variable_smap"] = IND.iloc[0].variable
V["rho_mejor_variable_smap"] = round(float(IND.iloc[0].spearman), 3)
print(f"    ICE vs S_MAP: rho = {V['rho_ice_smap']} (binaria {V['rho_binaria_smap']}); "
      f"mejor variable individual: {V['mejor_variable_smap']} ({V['rho_mejor_variable_smap']})")

f, ax = plt.subplots(1, 3, figsize=(16, 4.8))
ax[0].scatter(M.ICE, M.s_map, s=24, alpha=.75, c="#3b6ea5", edgecolor="none")
ax[0].set_xlabel("ICE"); ax[0].set_ylabel("$S_{MAP}$")
ax[0].set_title(f"(a) Indice continuo: $\\rho$ = {V['rho_ice_smap']:.3f}")
for c, nom, col in [(0, "grupo 'simple'", "#5b8ecb"), (1, "grupo 'complejo'", "#d98b5f")]:
    ax[1].hist(M.s_map[M.etiqueta_binaria == c], bins=14, alpha=.65, label=nom, color=col)
ax[1].set_xlabel("$S_{MAP}$"); ax[1].set_ylabel("piezas"); ax[1].legend(fontsize=8)
ax[1].set_title(f"(b) Etiqueta binaria: AUC = {V['auc_binaria']:.3f}")
d = TABLAS["T8_validacion_smap"]
ax[2].barh(range(len(d)), d.spearman.abs(), color="#3b6ea5")
ax[2].set_yticks(range(len(d)))
ax[2].set_yticklabels(["Binaria", "Promedio", "Cresta (CV)", "PC1 grafo", "ICE"], fontsize=9)
ax[2].set_xlabel("|Spearman| con $S_{MAP}$"); ax[2].set_title("(c) Comparacion")
f.suptitle(f"Figura 4. Validacion externa frente a la complejidad percibida (n = {len(M)})")
f.tight_layout(); fig("fig04_validacion_smap")


# =============================================================================
print("\n[5] Modelado de la calidad: modelo en dos partes")
# =============================================================================
y = df[C.OBJETIVO].values
y_bin = (y > 0).astype(int)
Xq = X.copy()
cv_c = StratifiedKFold(5, shuffle=True, random_state=C.SEMILLA)
cv_r = KFold(5, shuffle=True, random_state=C.SEMILLA)

# --- parte 1: ¿el modelo admite alguna modificacion?
modelos_c = {"Regresion logistica": LogisticRegression(max_iter=2000, random_state=C.SEMILLA),
             "Random Forest": RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                     random_state=C.SEMILLA, n_jobs=-1)}
fc = []
for nom, mod in modelos_c.items():
    p = cross_val_predict(mod, Xq, y_bin, cv=cv_c, method="predict_proba")[:, 1]
    fc.append({"modelo": nom, "AUC": roc_auc_score(y_bin, p),
               "exactitud": ((p > .5).astype(int) == y_bin).mean()})
TABLAS["T10_parte1_clasificacion"] = pd.DataFrame(fc)
mejor_c = max(fc, key=lambda r: r["AUC"])
V["auc_parte1"] = round(float(mejor_c["AUC"]), 3)
V["modelo_parte1"] = mejor_c["modelo"]
p_hat = cross_val_predict(modelos_c[mejor_c["modelo"]], Xq, y_bin, cv=cv_c,
                          method="predict_proba")[:, 1]

# --- parte 2: magnitud, solo en las piezas que regeneran
mask = y_bin == 1
from sklearn.linear_model import RidgeCV
modelos_r = {"Ridge (lineal)": RidgeCV(alphas=np.logspace(-3, 3, 25)),
             "Random Forest": RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                                    random_state=C.SEMILLA, n_jobs=-1)}
fr = []
for nom, mod in modelos_r.items():
    pr = cross_val_predict(mod, Xq[mask], y[mask], cv=cv_r)
    fr.append({"modelo": nom, "R2": r2_score(y[mask], pr), "MAE": mean_absolute_error(y[mask], pr)})
TABLAS["T11_parte2_regresion"] = pd.DataFrame(fr)
mejor_r = max(fr, key=lambda r: r["R2"])
V["r2_parte2"] = round(float(mejor_r["R2"]), 3)
V["mae_parte2"] = round(float(mejor_r["MAE"]), 3)
V["modelo_parte2"] = mejor_r["modelo"]

# --- comparacion: regresion ingenua sobre las 351 piezas (lo que hacia el borrador)
rf_all = RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                               random_state=C.SEMILLA, n_jobs=-1)
p_all = cross_val_predict(rf_all, Xq, y, cv=cv_r)
V["r2_ingenua"] = round(float(r2_score(y, p_all)), 3)
V["mae_ingenua"] = round(float(mean_absolute_error(y, p_all)), 3)

# --- prediccion combinada, enteramente fuera de muestra
# En cada particion se ajustan las dos partes SOLO con el tramo de entrenamiento
# y se predice el tramo de prueba, de modo que ninguna pieza influye en su propia
# prediccion (ni en la probabilidad ni en la magnitud).
esperado = np.zeros(len(y))
for tr, te in StratifiedKFold(5, shuffle=True, random_state=C.SEMILLA).split(Xq, y_bin):
    clf = RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                 random_state=C.SEMILLA, n_jobs=-1).fit(Xq[tr], y_bin[tr])
    tr_pos = tr[y_bin[tr] == 1]
    reg = RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                random_state=C.SEMILLA, n_jobs=-1).fit(Xq[tr_pos], y[tr_pos])
    esperado[te] = clf.predict_proba(Xq[te])[:, 1] * reg.predict(Xq[te])
mag = cross_val_predict(RandomForestRegressor(n_estimators=500, min_samples_leaf=3,
                                              random_state=C.SEMILLA, n_jobs=-1),
                        Xq[mask], y[mask], cv=cv_r)
V["r2_dos_partes"] = round(float(r2_score(y, esperado)), 3)
V["mae_dos_partes"] = round(float(mean_absolute_error(y, esperado)), 3)
TABLAS["T12_comparacion_calidad"] = pd.DataFrame([
    {"enfoque": "Regresion unica sobre las 351 piezas (borrador anterior)",
     "R2": V["r2_ingenua"], "MAE": V["mae_ingenua"]},
    {"enfoque": "Modelo en dos partes (probabilidad x magnitud)",
     "R2": V["r2_dos_partes"], "MAE": V["mae_dos_partes"]}])
print(f"    parte 1 AUC = {V['auc_parte1']} | parte 2 R2 = {V['r2_parte2']} | "
      f"combinado R2 = {V['r2_dos_partes']} (regresion unica {V['r2_ingenua']})")

f, ax = plt.subplots(1, 3, figsize=(16, 4.6))
ax[0].hist(y, bins=40, color="#3b6ea5")
ax[0].set_xlabel("success_ratio"); ax[0].set_ylabel("piezas")
ax[0].set_title(f"(a) El {V['pct_ceros']} % de las piezas vale 0")
fpr, tpr, _ = roc_curve(y_bin, p_hat)
ax[1].plot(fpr, tpr); ax[1].plot([0, 1], [0, 1], ls="--", c="gray")
ax[1].set_xlabel("1 - especificidad"); ax[1].set_ylabel("sensibilidad")
ax[1].set_title(f"(b) Parte 1: regenera / no regenera (AUC = {V['auc_parte1']:.3f})")
ax[2].scatter(y[mask], mag, s=16, alpha=.7, c="#3b6ea5", edgecolor="none")
lim = [0, max(y[mask].max(), mag.max())]
ax[2].plot(lim, lim, ls="--", c="gray")
ax[2].set_xlabel("observado"); ax[2].set_ylabel("predicho")
ax[2].set_title(f"(c) Parte 2: magnitud ($R^2$ = {V['r2_parte2']:.3f})")
f.suptitle("Figura 5. La calidad exige un modelo en dos partes")
f.tight_layout(); fig("fig05_calidad_dos_partes")


# =============================================================================
print("\n[6] Efecto de la fuente y relacion complejidad-calidad")
# =============================================================================
grandes = df.source.value_counts()[lambda s: s >= 10].index.tolist()
sub = df[df.source.isin(grandes)].copy()
V["n_fuentes_analizadas"] = len(grandes)
V["n_sub"] = len(sub)
fk = []
for v in C.VARS + [C.OBJETIVO]:
    g = [x[v].values for _, x in sub.groupby("source")]
    normal = all(shapiro(x).pvalue > .05 for x in g if len(x) >= 3)
    kw = kruskal(*g)
    eps2 = (kw.statistic - len(g) + 1) / (len(sub) - len(g))
    fk.append({"variable": v, "normalidad": normal,
               "homocedasticidad": levene(*g).pvalue > .05,
               "H": kw.statistic, "p": kw.pvalue, "epsilon2": eps2})
K = pd.DataFrame(fk).sort_values("epsilon2", ascending=False)
K["p_holm"] = np.minimum(1, K.p.values * (len(K) - np.arange(len(K))))
K["p_holm"] = K.p_holm.cummax()
TABLAS["T13_kruskal_fuente"] = K
V["n_normales"] = int(K.normalidad.sum())
V["n_significativas_fuente"] = int((K.p_holm < .05).sum())
V["eps2_max"] = round(float(K.epsilon2.max()), 3)
V["var_eps2_max"] = K.iloc[0].variable
V["p_fuente_sr"] = float(K.set_index("variable").loc[C.OBJETIVO, "p"])

rho_bruta = spearmanr(df.ICE, df[C.OBJETIVO])
V["rho_complejidad_calidad"] = round(float(rho_bruta.statistic), 3)
sub["r_i"] = rankdata(sub.ICE); sub["r_q"] = rankdata(sub[C.OBJETIVO])
D_src = sm.add_constant(pd.get_dummies(sub.source, drop_first=True).astype(float).values)
def _ols(yv, Xm):
    m = sm.OLS(np.asarray(yv, float), np.asarray(Xm, float)).fit()
    return m.resid, m.rsquared
e1, _ = _ols(sub.r_i, D_src)
e2, r2_src = _ols(sub.r_q, D_src)
pr = spearmanr(e1, e2)
V["rho_parcial"] = round(float(pr.statistic), 3)
V["p_parcial"] = float(pr.pvalue)
V["r2_fuente"] = round(float(r2_src), 3)
V["r2_ice"] = round(float(_ols(sub.r_q, sm.add_constant(sub[["r_i"]].values))[1]), 3)
V["r2_ambas"] = round(float(_ols(sub.r_q, np.column_stack([D_src, sub.r_i.values]))[1]), 3)
V["aporte_ice"] = round(100 * (V["r2_ambas"] - V["r2_fuente"]), 1)
TABLAS["T14_confusion_fuente"] = pd.DataFrame([
    {"modelo": "solo fuente", "R2_rangos": V["r2_fuente"]},
    {"modelo": "solo ICE", "R2_rangos": V["r2_ice"]},
    {"modelo": "ICE + fuente", "R2_rangos": V["r2_ambas"]}])
print(f"    rho(ICE, calidad) = {V['rho_complejidad_calidad']} -> parcial {V['rho_parcial']}")

f, ax = plt.subplots(1, 3, figsize=(16, 4.8))
q = TABLAS["A2_fuentes"].set_index("source").loc[grandes]
ax[0].barh(q.index, q.pct_cero, color="#c1666b")
ax[0].set_xlabel("% de piezas con success_ratio = 0")
ax[0].set_title("(a) Los ceros se concentran por origen")
top = K.variable.iloc[0]
ax[1].boxplot([sub[sub.source == s][top].values for s in grandes],
              tick_labels=[g[:11] for g in grandes], showfliers=False)
ax[1].tick_params(axis="x", rotation=45, labelsize=7)
ax[1].set_title(f"(b) {top} por fuente ($\\varepsilon^2$ = {K.iloc[0].epsilon2:.3f})")
ax[2].scatter(df.ICE, df[C.OBJETIVO], s=14, alpha=.6, c="#3b6ea5", edgecolor="none")
ax[2].set_xlabel("ICE"); ax[2].set_ylabel("success_ratio")
ax[2].set_title(f"(c) $\\rho$ = {V['rho_complejidad_calidad']:.3f}; "
                f"parcial = {V['rho_parcial']:.3f}")
f.suptitle("Figura 6. La fuente del modelo condiciona metricas y calidad")
f.tight_layout(); fig("fig06_fuente_y_calidad")


# =============================================================================
print("\n[7] Salidas")
# =============================================================================
df[["file_name", "source", "id"] + C.VARS + [C.OBJETIVO, "ICE", "etiqueta_binaria"]] \
    .to_excel(f"{C.RES}/dataset_procesado.xlsx", index=False)
with pd.ExcelWriter(f"{C.RES}/tablas.xlsx") as w:
    for nom, t in TABLAS.items():
        t.to_excel(w, sheet_name=nom[:31], index=False)
V["_generado"] = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
with open(f"{C.RES}/valores_paper.json", "w", encoding="utf-8") as fh:
    json.dump({k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
               for k, v in V.items()}, fh, indent=2, ensure_ascii=False)
print(f"    tablas.xlsx ({len(TABLAS)} hojas), valores_paper.json, dataset_procesado.xlsx")
print(f"    figuras en {C.FIG}")
print("\nListo.")
