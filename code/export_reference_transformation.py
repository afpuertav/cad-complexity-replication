# -*- coding: utf-8 -*-
"""Exports the reference transformation that defines ICE (Structural Complexity Index).

ICE is defined relative to the 351-model reference corpus: a quantile->normal transform and a
standardisation of the 10 retained metrics, followed by the first principal component.
This script saves every fitted quantity so ICE can be computed for new models:

    z   = (Phi^-1(F_j(x_j)) - mean_j) / scale_j          (F_j = empirical CDF of metric j, interpolated)
    ICE = sign * sum_j loading_j * z_j
"""
import os, sys, json
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler, QuantileTransformer
from sklearn.decomposition import PCA
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import config as C

raw = pd.read_excel(C.ARCHIVO_DATOS)
agg = {c: "mean" for c in raw.columns if c not in ("file_name", "source")}
agg["source"] = "first"
df = raw.groupby("file_name", as_index=False).agg(agg)          # 352 records -> 351 models
X0 = df[C.VARS]
Q = QuantileTransformer(output_distribution="normal", n_quantiles=len(df), random_state=C.SEMILLA).fit(X0)
S = StandardScaler().fit(Q.transform(X0))
Z = S.transform(Q.transform(X0))
pca = PCA(n_components=1, random_state=C.SEMILLA).fit(Z)
sign = float(np.sign(spearmanr(pca.transform(Z)[:, 0], df.cyclomatic_complexity).statistic))
out = {"variables": C.VARS, "n_models": int(len(df)), "seed": C.SEMILLA,
       "quantiles_reference": Q.quantiles_.tolist(), "references": Q.references_.tolist(),
       "scaler_mean": S.mean_.tolist(), "scaler_scale": S.scale_.tolist(),
       "pc1_loadings": (pca.components_[0] * sign).tolist(), "pc1_mean": pca.mean_.tolist(),
       "explained_variance_ratio_pc1": float(pca.explained_variance_ratio_[0])}
os.makedirs(C.RES, exist_ok=True)
json.dump(out, open(os.path.join(C.RES, "ice_reference_transformation.json"), "w"), indent=1)
print("saved; PC1 explains %.3f of the variance" % out["explained_variance_ratio_pc1"])
