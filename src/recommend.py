"""Recomendação personalizada de exercícios (baseada em conteúdo).

Recomenda novas questões a partir do histórico do aluno (questões já resolvidas)
e, opcionalmente, do nível de dificuldade desejado: o perfil do aluno é o
centroide dos vetores TF-IDF das questões que ele resolveu, e recomendamos as
questões não resolvidas mais similares (similaridade do cosseno).

Por ser baseado em **conteúdo** (e não em rótulo), o recomendador combina várias
fontes num só catálogo — inclusive as **não rotuladas** (SPOJ, OBI), que não
servem ao classificador mas são candidatas válidas aqui. As fontes vêm de
``config.RECOMMENDER_SOURCES`` (no ``.env``); vazio -> usa só a fonte ativa.
O filtro por nível só se aplica às questões rotuladas (INF110, Neps).

Pré-requisito: cada fonte usada precisa ter sido consolidada antes pela Etapa 1
(``DATASET=<fonte> python -m src.ingest``).

Uso (demonstração):
    python -m src.recommend
    RECOMMENDER_SOURCES=INF110,Neps,SPOJ,OBI python -m src.recommend
"""
from __future__ import annotations

import argparse
import unicodedata

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config, data_utils

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
    """Recomendador baseado em conteúdo (TF-IDF + similaridade do cosseno).

    Combina as fontes em ``fontes`` (padrão: ``config.RECOMMENDER_SOURCES`` ou,
    se vazio, só a fonte ativa) num catálogo único e ajusta um TF-IDF sobre todo
    ele, de modo que questões de qualquer fonte possam ser recomendadas.
    """

    def __init__(
        self,
        fontes: list[str] | None = None,
        max_features: int = 5000,
        ngram_max: int = 2,
        min_df: int = 2,
    ) -> None:
        self.fontes = fontes or config.RECOMMENDER_SOURCES or [config.DATASET]
        self.df = self._carregar_catalogo(self.fontes)

        # Mesma limpeza usada no pré-processamento, para consistência.
        self.df["texto_limpo"] = self.df[config.TEXT_COL].astype(str).map(data_utils.clean_text)
        self.df = self.df[self.df["texto_limpo"].str.len() > 0].reset_index(drop=True)

        # TF-IDF ajustado sobre o catálogo combinado (vocabulário cobre todas as
        # fontes). É independente do vetorizador do classificador.
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, ngram_max), min_df=min_df
        )
        self.X = self.vectorizer.fit_transform(self.df["texto_limpo"])

    def _carregar_catalogo(self, fontes: list[str]) -> pd.DataFrame:
        """Lê e concatena o questoes.csv de cada fonte, marcando a origem."""
        partes: list[pd.DataFrame] = []
        for fonte in fontes:
            csv = config.questoes_csv_for(fonte)
            if not csv.exists():
                raise FileNotFoundError(
                    f"Base da fonte '{fonte}' não encontrada em {csv}. "
                    f"Rode antes: DATASET={fonte} python -m src.ingest"
                )
            df = pd.read_csv(csv)
            if config.LABEL_COL not in df.columns:
                df[config.LABEL_COL] = pd.NA
            df = df[[config.ID_COL, config.TEXT_COL, config.LABEL_COL]].copy()
            df["fonte"] = fonte
            partes.append(df)
            rotuladas = df[config.LABEL_COL].notna().sum()
            print(f"[{fonte}] {len(df)} questões ({rotuladas} rotuladas)")
        catalogo = pd.concat(partes, ignore_index=True)
        print(f"Catálogo combinado: {len(catalogo)} questões de {len(fontes)} fonte(s)\n")
        return catalogo

    def recomendar(
        self,
        resolvidos_idx: list[int],
        nivel_alvo: str | None = None,
        top_k: int = 5,
    ) -> pd.DataFrame:
        """Recomenda até top_k questões não resolvidas.

        resolvidos_idx : índices (linhas do catálogo) já resolvidos pelo aluno.
        nivel_alvo     : se informado, filtra por esse nível (ex.: 'medio'); as
                         questões sem rótulo (SPOJ/OBI) ficam de fora do filtro.
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

        colunas = [c for c in (config.ID_COL, "fonte", config.TEXT_COL, config.LABEL_COL) if c in df.columns]
        colunas.append("similaridade")
        return (
            df[mask]
            .sort_values("similaridade", ascending=False)
            .head(top_k)[colunas]
            .reset_index(names="indice")
        )

    def recomendar_por_texto(
        self,
        texto: str,
        nivel_alvo: str | None = None,
        top_k: int = 5,
        excluir_idx: list[int] | None = None,
    ) -> pd.DataFrame:
        """Recomenda questões do catálogo mais similares a um TEXTO arbitrário.

        Diferente de ``recomendar`` (que parte de índices já no catálogo), aqui o
        enunciado pode ser de uma questão NOVA (ex.: a questão que o professor
        quer avaliar). O texto é limpo e projetado no mesmo espaço TF-IDF do
        catálogo; recomendamos as questões mais similares (cosseno).

        excluir_idx : índices do catálogo a omitir (ex.: a própria questão, se ela
                      já estiver no catálogo, para não se recomendar a si mesma).
        """
        vetor = self.vectorizer.transform([data_utils.clean_text(str(texto))])
        df = self.df.copy()
        df["similaridade"] = cosine_similarity(vetor, self.X).ravel()

        mask = pd.Series(True, index=df.index)
        if excluir_idx:
            mask &= ~df.index.isin(excluir_idx)
        if nivel_alvo and config.LABEL_COL in df.columns:
            alvo = _norm_nivel(nivel_alvo)
            mask &= df[config.LABEL_COL].map(_norm_nivel) == alvo

        colunas = [c for c in (config.ID_COL, "fonte", config.TEXT_COL, config.LABEL_COL) if c in df.columns]
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
        niveis = niveis[niveis != "nan"]
        nivel_atual = niveis.mode().iloc[0] if len(niveis) else "facil"
        alvo = PROXIMO_NIVEL.get(nivel_atual, nivel_atual)
        print(f"Nível predominante do aluno: {nivel_atual} -> recomendando: {alvo}")
        return self.recomendar(resolvidos_idx, nivel_alvo=alvo, top_k=top_k)


def _resumo_enunciado(texto: str, n: int = 80) -> str:
    txt = " ".join(str(texto).split())
    return txt[:n] + ("…" if len(txt) > n else "")


def run(top_k: int = 5) -> None:
    rec = RecomendadorConteudo()
    print(f"Fontes no catálogo: {', '.join(rec.fontes)}\n")

    # Demonstração: simula um aluno que resolveu algumas questões fáceis.
    if config.LABEL_COL in rec.df.columns and rec.df[config.LABEL_COL].notna().any():
        faceis = rec.df[rec.df[config.LABEL_COL].map(_norm_nivel) == "facil"]
        resolvidos = faceis.sample(min(3, len(faceis)), random_state=config.RANDOM_SEED).index.tolist()
    else:
        resolvidos = rec.df.sample(min(3, len(rec.df)), random_state=config.RANDOM_SEED).index.tolist()

    print(f"Aluno (demo) resolveu os índices: {resolvidos}\n")

    def _mostrar(df: pd.DataFrame) -> None:
        if config.TEXT_COL in df.columns:
            df = df.assign(**{config.TEXT_COL: df[config.TEXT_COL].map(_resumo_enunciado)})
        print(df.to_string(index=False))

    print(f"=== Recomendações por nível (próximo passo, top {top_k}) ===")
    _mostrar(rec.recomendar_proximo_nivel(resolvidos, top_k=top_k))

    # Sem filtro de nível: percorre todo o catálogo, então questões sem rótulo
    # (SPOJ, OBI) também podem aparecer.
    print(f"\n=== Recomendações por conteúdo (qualquer fonte, top {top_k}) ===")
    _mostrar(rec.recomendar(resolvidos, top_k=top_k))


def main() -> None:
    parser = argparse.ArgumentParser(description="Recomendador de exercícios (demo)")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()
    run(top_k=args.top_k)


if __name__ == "__main__":
    main()
