# -*- coding: utf-8 -*-
"""Rutas y conjuntos de variables del pipeline del articulo (version 2026-09-05)."""
import os

AQUI = os.path.dirname(os.path.abspath(__file__))
BASE = os.path.dirname(AQUI)                           # deposit root (parent of code/)
DATOS = os.path.join(BASE, "data")
RES   = os.path.join(BASE, "results")
FIG   = os.path.join(BASE, "results", "figures")
for d in (RES, FIG):
    os.makedirs(d, exist_ok=True)

ARCHIVO_DATOS = os.path.join(DATOS, "Dataset_complejidad_con_size.xlsx")
ARCHIVO_SMAP  = os.path.join(DATOS, "resultados_106.xlsx")
HOJA_SMAP     = "solver original"

# --- Trazabilidad de variables (responde al comentario C31) -------------------
# 19 predictores extraidos; 3 quedan inservibles por un fallo de extraccion.
CORRUPTAS = ["n_dimensions", "is_surface_model", "n_dependencies"]
# Descartadas por redundancia (se justifica con numeros en la seccion 2 del pipeline)
REDUNDANTES = {
    "kolmogorov_cl": "entropy_wang",
    "cube_ratio":    "sphere_ratio",
    "volume":        "volume_area_ratio",
    "surface":       "volume_area_ratio",
    "mass":          "volume",          # densidad por defecto: mass = volume x 1000
}
# Descartada por medir el proceso y no el modelo (C35)
PROCESO = ["time"]

GRAFO = ["cyclomatic_complexity", "modularity", "graph_density",
         "entropy_wang", "entropy_infoshannon", "entropy_shannon"]
GEOMETRIA = ["volume_area_ratio", "volume_ratio", "sphere_ratio"]
ARCHIVO = ["size"]
VARS = GRAFO + GEOMETRIA + ARCHIVO          # 10 variables del estudio
OBJETIVO = "success_ratio"

FAMILIA = ({v: "grafo" for v in GRAFO} | {v: "geometrica" for v in GEOMETRIA}
           | {v: "archivo" for v in ARCHIVO})

SEMILLA = 20260905
