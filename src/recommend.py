"""Recomendação personalizada de exercícios.

Recomenda novas questões com base no histórico do aluno (questões já
resolvidas) e no nível de dificuldade desejado, usando filtragem baseada em
conteúdo: o perfil do aluno é o centroide dos vetores TF-IDF das questões que
ele resolveu, e recomendamos as questões não resolvidas mais similares.

Pré-requisito: rodar antes `python -m src.preprocess`.

Uso (demonstração):
    python -m src.recommend
"""
from __future__ import annotations

import argparse
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from . import config

# Progressão sugerida de nível (para recomendar o "próximo passo").
PROXIMO_NIVEL = {
    "muito_facil": "facil",
    "facil": "medio",
    "medio": "dificil",
    "dificil": "muito_dificil",
    "muito_dificil": "muito_dificil",
}


def _norm_nivel(nivel: str) -> str:
    txt = unicodedata.normalize("NFKD", str(nivel).lower().strip())
    return "".join(c for c in txt if not unicodedata.combining(c))


class RecomendadorConteudo:
    """Recomendador baseado em conteúdo (TF-IDF + similaridade do cosseno)."""

    def __init__(self) -> None:
        if not config.CLEAN_DATASET.exists():
            raise FileNotFoundError(
                "dataset_limpo.csv não encontrado. Rode antes: python -m src.preprocess"
            )
        self.vectorizer = joblib.load(config.TFIDF_VECTORIZER)
        self.df = pd.read_csv(config.CLEAN_DATASET).reset_index(drop=True)
        self.X = self.vectorizer.transform(self.df["texto_limpo"].astype(str))

    def recomendar(
        self,
        resolvidos_idx: list[int],
        nivel_alvo: str | None = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Recomenda até top_k questões não resolvidas.

        resolvidos_idx : índices (linhas do dataset_limpo) já resolvidos pelo aluno.
        nivel_alvo     : se informado, filtra recomendações por esse nível
                         (ex.: 'medio'); caso contrário, considera todas.
        """
        df = self.df.copy()

        if resolvidos_idx:
            perfil = np.asarray(self.X[resolvidos_idx].mean(axis=0)).reshape(1, -1)
            df["similaridade"] = cosine_similarity(perfil, self.X).ravel()
        else:
            # Aluno sem histórico: recomenda por nível, sem ranqueamento de similaridade.
            df["similaridade"] = 0.0

        mask = ~df.index.isin(resolvidos_idx)
        if nivel_alvo and config.LABEL_COL in df.columns:
            alvo = _norm_nivel(nivel_alvo)
            mask &= df[config.LABEL_COL].map(_norm_nivel) == alvo

        colunas = [c for c in (config.ID_COL, config.TEXT_COL, config.LABEL_COL) if c in df.columns]
        colunas.append("similaridade")
        return (
            df[mask]
            .sort_values("similaridade", ascending=False)
            .head(top_k)[colunas]
            .reset_index(names="indice")
        )

    def recomendar_proximo_nivel(self, resolvidos_idx: list[int], top_k: int = 5) -> pd.DataFrame:
        """Recomenda no nível imediatamente acima do que o aluno mais resolveu."""
        if not resolvidos_idx or config.LABEL_COL not in self.df.columns:
            return self.recomendar(resolvidos_idx, top_k=top_k)
        niveis = self.df.loc[resolvidos_idx, config.LABEL_COL].map(_norm_nivel)
        nivel_atual = niveis.mode().iloc[0] if len(niveis) else "facil"
        alvo = PROXIMO_NIVEL.get(nivel_atual, nivel_atual)
        print(f"Nível predominante do aluno: {nivel_atual} -> recomendando: {alvo}")
        return self.recomendar(resolvidos_idx, nivel_alvo=alvo, top_k=top_k)


def run(top_k: int = 5) -> None:
    rec = RecomendadorConteudo()
    # Demonstração: simula um aluno que resolveu algumas questões fáceis.
    if config.LABEL_COL in rec.df.columns:
        faceis = rec.df[rec.df[config.LABEL_COL].map(_norm_nivel) == "facil"]
        resolvidos = faceis.sample(min(3, len(faceis)), random_state=config.RANDOM_SEED).index.tolist()
    else:
        resolvidos = rec.df.sample(min(3, len(rec.df)), random_state=config.RANDOM_SEED).index.tolist()

    print(f"Aluno (demo) resolveu os índices: {resolvidos}\n")
    print(f"=== Recomendações (próximo nível, top {top_k}) ===")
    print(rec.recomendar_proximo_nivel(resolvidos, top_k=top_k).to_string(index=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recomendador de exercícios (demo)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(top_k=args.top_k)


if __name__ == "__main__":
    main()
