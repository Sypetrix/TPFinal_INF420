"""Gera as figuras do relatório a partir dos artefatos salvos pela Etapa 3.

Lê ``models/<DATASET>/ml_metrics.csv``, ``ml_metrics_3niveis.csv`` e
``matriz_confusao.csv`` e produz, na pasta ``figuras/`` (versionada, para
acompanhar o ``main.tex``):

  - ``matriz_confusao.png`` : mapa de calor da matriz de confusão (5 níveis, KNN)
  - ``comparacao_f1.png``   : F1-macro por modelo, 5 vs 3 níveis

Pré-requisito: rodar antes ``python -m src.train_ml`` (5 níveis) e
``python -m src.train_ml --niveis 3`` (análise complementar).

Uso:
    python -m src.figuras
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # backend sem display, para salvar arquivos

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from . import config  # noqa: E402

FIG_DIR = config.ROOT_DIR / "figuras"

# Abreviações dos níveis para caber nos eixos das figuras.
ABREV = {
    "muito_facil": "MF", "facil": "F", "medio": "M",
    "dificil": "D", "muito_dificil": "MD",
}
MODELOS = ["KNN", "Regressao Logistica", "SVM (linear)", "Random Forest"]
MODELOS_CURTO = ["KNN", "Reg. Log.", "SVM", "RF"]


def _heatmap_confusao() -> None:
    cm = pd.read_csv(config.MODELS_DIR / "matriz_confusao.csv", index_col=0)
    rotulos = [ABREV.get(c, c) for c in cm.index]
    valores = cm.values

    fig, ax = plt.subplots(figsize=(4.0, 3.2))
    im = ax.imshow(valores, cmap="Blues")
    ax.set_xticks(range(len(rotulos)), labels=rotulos)
    ax.set_yticks(range(len(rotulos)), labels=rotulos)
    ax.set_xlabel("Classe prevista")
    ax.set_ylabel("Classe verdadeira")
    limiar = valores.max() / 2 if valores.max() else 1
    for i in range(valores.shape[0]):
        for j in range(valores.shape[1]):
            ax.text(j, i, int(valores[i, j]), ha="center", va="center",
                    color="white" if valores[i, j] > limiar else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    saida = FIG_DIR / "matriz_confusao.png"
    fig.savefig(saida, dpi=200)
    plt.close(fig)
    print("salvo:", saida)


def _f1_por_modelo(df: pd.DataFrame) -> list[float]:
    return [
        float(df.loc[df["modelo"] == nome, "f1_macro"].iloc[0])
        if (df["modelo"] == nome).any() else np.nan
        for nome in MODELOS
    ]


def _barras_f1() -> None:
    m5 = pd.read_csv(config.MODELS_DIR / "ml_metrics.csv")
    m3 = pd.read_csv(config.MODELS_DIR / "ml_metrics_3niveis.csv")
    v5, v3 = _f1_por_modelo(m5), _f1_por_modelo(m3)

    x = np.arange(len(MODELOS))
    largura = 0.38
    fig, ax = plt.subplots(figsize=(6.0, 3.3))
    ax.bar(x - largura / 2, v5, largura, label="5 níveis")
    ax.bar(x + largura / 2, v3, largura, label="3 níveis")
    ax.set_xticks(x, labels=MODELOS_CURTO)
    ax.set_ylabel("F1-macro")
    ax.set_ylim(0, 0.6)
    ax.legend()
    for i, (a, b) in enumerate(zip(v5, v3)):
        ax.text(i - largura / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=8)
        ax.text(i + largura / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    saida = FIG_DIR / "comparacao_f1.png"
    fig.savefig(saida, dpi=200)
    plt.close(fig)
    print("salvo:", saida)


def run() -> None:
    FIG_DIR.mkdir(exist_ok=True)
    _heatmap_confusao()
    _barras_f1()


def main() -> None:
    run()


if __name__ == "__main__":
    main()
