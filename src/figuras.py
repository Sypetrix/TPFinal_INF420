"""Gera as figuras do relatorio a partir dos artefatos salvos pela Etapa 3."""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config

FIG_DIR = config.ROOT_DIR / "figuras"
ABREV = {"muito_facil": "MF", "facil": "F", "medio": "M", "dificil": "D", "muito_dificil": "MD"}
MODELOS = ["KNN", "Regressao Logistica", "SVM (linear)", "Random Forest"]
MODELOS_CURTO = ["KNN", "Reg. Log.", "SVM", "RF"]


def _f1_por_modelo(df):
    return [float(df.loc[df["modelo"] == n, "f1_macro"].iloc[0]) if (df["modelo"] == n).any() else np.nan for n in MODELOS]


def _painel_matrizes():
    MODELS = config.ROOT_DIR / "models"
    fontes = ["INF110", "Neps"]
    csvs = {}
    for f in fontes:
        c5 = MODELS / f / "matriz_confusao.csv"
        c3 = MODELS / f / "matriz_confusao_3niveis.csv"
        if not (c5.exists() and c3.exists()):
            print(f"[aviso] artefatos de {f} faltando; painel NAO gerado."); return
        csvs[f] = (c5, c3)
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.0))
    titulos = [("INF110 - 5 niveis", csvs["INF110"][0]),
               ("INF110 - 3 niveis", csvs["INF110"][1]),
               ("Neps - 5 niveis", csvs["Neps"][0]),
               ("Neps - 3 niveis", csvs["Neps"][1])]
    for ax, (t, p) in zip(axes.flat, titulos):
        cm = pd.read_csv(p, index_col=0)
        rot = [ABREV.get(c, c) for c in cm.index]
        v = cm.values
        im = ax.imshow(v, cmap="Blues")
        ax.set_xticks(range(len(rot)), labels=rot); ax.set_yticks(range(len(rot)), labels=rot)
        ax.set_title(t, fontsize=11); ax.set_xlabel("Prevista", fontsize=9); ax.set_ylabel("Verdadeira", fontsize=9)
        lim = v.max() / 2 if v.max() else 1
        for i in range(v.shape[0]):
            for j in range(v.shape[1]):
                ax.text(j, i, int(v[i, j]), ha="center", va="center",
                        color="white" if v[i, j] > lim else "black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    s = FIG_DIR / "matrizes_consolidadas.png"
    fig.savefig(s, dpi=200); plt.close(fig); print("salvo:", s)


def _barras_f1_consolidadas():
    MODELS = config.ROOT_DIR / "models"
    fontes = ["INF110", "Neps"]
    dados = {}
    for f in fontes:
        m5 = MODELS / f / "ml_metrics.csv"; m3 = MODELS / f / "ml_metrics_3niveis.csv"
        if not (m5.exists() and m3.exists()):
            print(f"[aviso] metricas de {f} faltando; NAO gerado."); return
        dados[f] = (pd.read_csv(m5), pd.read_csv(m3))
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.6), sharey=True)
    x = np.arange(len(MODELOS)); w = 0.38
    for ax, f in zip(axes, fontes):
        m5, m3 = dados[f]; v5, v3 = _f1_por_modelo(m5), _f1_por_modelo(m3)
        ax.bar(x - w/2, v5, w, label="5 niveis"); ax.bar(x + w/2, v3, w, label="3 niveis")
        ax.set_xticks(x, labels=MODELOS_CURTO); ax.set_title(f, fontsize=11); ax.set_ylim(0, 0.75)
        for i, (a, b) in enumerate(zip(v5, v3)):
            ax.text(i - w/2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
            ax.text(i + w/2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    axes[0].set_ylabel("F1-macro"); axes[0].legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    s = FIG_DIR / "comparacao_f1_consolidada.png"
    fig.savefig(s, dpi=200); plt.close(fig); print("salvo:", s)


def run():
    FIG_DIR.mkdir(exist_ok=True)
    _painel_matrizes()
    _barras_f1_consolidadas()


def main():
    run()


if __name__ == "__main__":
    main()
