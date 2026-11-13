# -*- coding: utf-8 -*-
"""Figura 1: flujo metodologico del estudio.

Se genera aparte del pipeline porque es un diagrama, no un resultado: no
depende de los datos y no cambia al reejecutar el analisis. Salida en
05_figuras/paper20260905, con el mismo formato que el resto de figuras.

    python figura_flujo.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import config as C

BORDE, TEXTO, TENUE = "#555555", "#111111", "#777777"

ETAPAS = [
    ("1", "Extracción de métricas",
     "Diecinueve métricas candidatas\nsobre 351 piezas de SolidWorks", "§ 3.2 · tabla 1"),
    ("2", "Auditoría y depuración",
     "Descarte por invalidez y por\nredundancia: de 19 a 10 variables", "§ 3.3 · tablas 2 y 3"),
    ("3", "Estructura de dependencia",
     "Spearman y D de Hoeffding\nsobre los 55 pares de variables", "§ 3.4 · figura 3"),
    ("4", "Estructura del espacio e índice",
     "Silueta frente a un nulo, gap y ARI;\nel ICE como primera componente", "§ 3.4–3.5 · figura 4"),
    ("5", "Validación externa",
     "Contraste del índice frente a la\ncomplejidad percibida (n = 106)", "§ 4.3 · tabla 4 · figura 5"),
    ("6", "Calidad y efecto de la fuente",
     "Modelo en dos partes y control\npor origen del modelo", "§ 3.6–3.7 · tablas 6 y 7"),
]

AN, AL = 4.30, 2.05          # ancho y alto de cada caja, en unidades de datos
COL, FIL = [0.6, 5.5, 10.4], [3.30, 0.15]
CANAL = 2.78                 # banda libre entre las dos filas, para el retorno

fig, ax = plt.subplots(figsize=(15.6, 5.6))
ax.set_xlim(-2.45, 18.3); ax.set_ylim(-0.35, 5.75); ax.axis("off")

centros = []
for k, (num, titulo, detalle, ref) in enumerate(ETAPAS):
    x, y = COL[k % 3], FIL[k // 3]
    ax.add_patch(FancyBboxPatch((x, y), AN, AL, boxstyle="round,pad=0.02,rounding_size=0.12",
                                linewidth=0.9, edgecolor=BORDE, facecolor="none"))
    ax.text(x + 0.22, y + AL - 0.42, num, fontsize=15, color=TENUE, va="center", ha="center")
    ax.text(x + 0.52, y + AL - 0.42, titulo, fontsize=11, color=TEXTO, va="center", weight="bold")
    ax.text(x + 0.52, y + AL - 1.02, detalle, fontsize=9.2, color=TEXTO, va="top", linespacing=1.45)
    ax.text(x + 0.52, y + 0.22, ref, fontsize=8.2, color=TENUE, va="center", style="italic")
    centros.append((x, y))

def flecha(p1, p2):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                                 linewidth=0.9, color=BORDE, shrinkA=0, shrinkB=0))

for k in (0, 1, 3, 4):                       # avance dentro de cada fila
    x, y = centros[k]
    flecha((x + AN + 0.06, y + AL / 2), (COL[(k % 3) + 1] - 0.06, y + AL / 2))

# retorno de la etapa 3 a la etapa 4, por la banda libre entre filas
xa = COL[2] + AN / 2
xb = COL[0] + AN / 2
ax.plot([xa, xa, xb], [FIL[0] - 0.06, CANAL, CANAL], color=BORDE, linewidth=0.9,
        solid_joinstyle="miter", clip_on=False)
flecha((xb, CANAL), (xb, FIL[1] + AL + 0.06))

# entrada y salida
ax.text(-0.35, FIL[0] + AL / 2, "corpus CAD\ny escala $S_{MAP}$", fontsize=9.2, color=TENUE,
        ha="right", va="center", linespacing=1.5)
flecha((-0.28, FIL[0] + AL / 2), (COL[0] - 0.06, FIL[0] + AL / 2))
flecha((COL[2] + AN + 0.06, FIL[1] + AL / 2), (COL[2] + AN + 0.55, FIL[1] + AL / 2))
ax.text(COL[2] + AN + 0.68, FIL[1] + AL / 2,
        "índice continuo,\nmodelo de calidad\ny cifras del artículo",
        fontsize=9.2, color=TENUE, ha="left", va="center", linespacing=1.5)

fig.tight_layout()
for ext in ("png", "pdf"):
    fig.savefig(f"{C.FIG}/fig00_flujo_metodologico.{ext}", dpi=160, bbox_inches="tight")
print("escrita fig00_flujo_metodologico en", C.FIG)
