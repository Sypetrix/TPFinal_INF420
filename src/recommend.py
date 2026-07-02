"""Recomendação de exercícios (por conceitos + dificuldade).

Conforme o contexto do projeto, a recomendação parte das questões avaliadas e
busca, no catálogo (a "base de recomendação"), questões **com conceitos em comum**
e **dificuldade compatível** (mesmo nível ou um acima/abaixo).

Pontos-chave:
  * **Critério de similaridade = conceitos** (extraídos pela LLM, ver
    ``llm_concepts``). Quando os conceitos do catálogo não estão disponíveis, o
    recomendador **cai para similaridade TF-IDF** do enunciado (degradação suave).
  * **Toda a base ganha uma dificuldade**: questões sem rótulo têm a dificuldade
    **predita pelo modelo de ML treinado** (na fonte ativa, ``DATASET``), para que
    o filtro por nível valha para o catálogo inteiro (contexto.md).
  * Opera em **3 níveis** (padrão, fácil/médio/difícil) ou **5 níveis**.

Catálogo combinado a partir de ``config.RECOMMENDER_SOURCES`` (no ``.env``); cada
fonte precisa ter sido consolidada antes (``DATASET=<fonte> python -m src.ingest``).

Uso (demonstração com histórico de aluno):
    python -m src.recommend
    python -m src.recommend --niveis 5
"""
from __future__ import annotations

import argparse
import unicodedata

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from . import config, data_utils, llm_concepts

# Peso dos conceitos no escore combinado (o restante vai para o TF-IDF). Os
# conceitos são o critério principal; o TF-IDF desempata e cobre quando faltam.
PESO_CONCEITOS = 0.7


def _norm_nivel(nivel: str) -> str:
    txt = unicodedata.normalize("NFKD", str(nivel).lower().strip())
    return "".join(c for c in txt if not unicodedata.combining(c))


class RecomendadorConteudo:
    """Recomendador por conceitos + dificuldade (com fallback TF-IDF)."""

    def __init__(
        self,
        fontes: list[str] | None = None,
        niveis: int = 3,
        prever_dificuldade: bool = True,
        max_features: int = 5000,
        ngram_max: int = 2,
        min_df: int = 2,
    ) -> None:
        self.fontes = fontes or config.RECOMMENDER_SOURCES or [config.DATASET]
        self.niveis = niveis
        self.ordem = config.DIFFICULTY_LABELS_3 if niveis == 3 else config.DIFFICULTY_LABELS
        self.idx_nivel = {n: i for i, n in enumerate(self.ordem)}

        self.df = self._carregar_catalogo(self.fontes)
        self.df["texto_limpo"] = self.df[config.TEXT_COL].astype(str).map(data_utils.clean_text)
        self.df = self.df[self.df["texto_limpo"].str.len() > 0].reset_index(drop=True)

        # Rótulo na escala escolhida (5 níveis -> colapsa para 3 quando niveis=3).
        rot = self.df[config.LABEL_COL].astype("string").str.lower()
        if niveis == 3:
            rot = rot.map(lambda v: config.COLLAPSE_5_TO_3.get(v, v) if pd.notna(v) else v)
        self.df[config.LABEL_COL] = rot

        # TF-IDF do catálogo combinado (fallback de similaridade).
        self.vectorizer = TfidfVectorizer(
            max_features=max_features, ngram_range=(1, ngram_max), min_df=min_df
        )
        self.X = self.vectorizer.fit_transform(self.df["texto_limpo"])

        # Conceitos por questão (critério principal), se houver cache por fonte.
        mapa = llm_concepts.carregar_conceitos(self.fontes)
        self.tem_conceitos = bool(mapa)
        self.conceitos = [
            mapa.get((row["fonte"], str(row[config.ID_COL])), set())
            for _, row in self.df.iterrows()
        ]

        # Dificuldade efetiva: rótulo quando existe; senão, predita pelo modelo de ML.
        self.df["dificuldade_efetiva"] = self.df[config.LABEL_COL]
        if prever_dificuldade:
            self._preencher_dificuldade_predita()

    # ------------------------------------------------------------------ catálogo
    def _carregar_catalogo(self, fontes: list[str]) -> pd.DataFrame:
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

    def _modelo_path(self):
        nome = "best_ml_model_3niveis.joblib" if self.niveis == 3 else "best_ml_model.joblib"
        return config.MODELS_DIR / nome

    def _preencher_dificuldade_predita(self) -> None:
        """Prediz a dificuldade das questões SEM rótulo com o modelo treinado."""
        sem_rotulo = self.df["dificuldade_efetiva"].isna()
        if not sem_rotulo.any():
            return
        modelo_path = self._modelo_path()
        if not modelo_path.exists():
            print(
                f"[aviso] modelo {modelo_path.name} não encontrado em {config.MODELS_DIR}; "
                "questões sem rótulo ficam sem dificuldade (treine com "
                f"python -m src.train_ml{' --niveis 3' if self.niveis == 3 else ''})."
            )
            return
        modelo = joblib.load(modelo_path)
        preds = modelo.predict(self.df.loc[sem_rotulo, "texto_limpo"].astype(str))
        self.df.loc[sem_rotulo, "dificuldade_efetiva"] = [str(p) for p in preds]
        print(f"Dificuldade predita para {int(sem_rotulo.sum())} questão(ões) sem rótulo "
              f"(modelo {modelo_path.name}).")

    # ------------------------------------------------------------------ scores
    def _niveis_aceitos(self, nivel_alvo: str | None, janela: int) -> set[str] | None:
        """Conjunto de níveis dentro de ±janela do alvo (None = sem filtro)."""
        if not nivel_alvo:
            return None
        i = self.idx_nivel.get(_norm_nivel(nivel_alvo))
        if i is None:
            return None
        return {self.ordem[j] for j in range(max(0, i - janela), min(len(self.ordem), i + janela + 1))}

    def _cosseno(self, vetor_tfidf) -> np.ndarray:
        return cosine_similarity(vetor_tfidf, self.X).ravel()

    def _score_vec(self, cos: np.ndarray, conceitos_query: set[str] | None) -> np.ndarray:
        """Escore por linha a partir do cosseno TF-IDF + conceitos (principal)."""
        if self.tem_conceitos and conceitos_query:
            jac = np.array([
                (len(conceitos_query & c) / len(conceitos_query | c)) if (conceitos_query | c) else 0.0
                for c in self.conceitos
            ])
            return PESO_CONCEITOS * jac + (1 - PESO_CONCEITOS) * cos
        return cos

    def _saida(self, df: pd.DataFrame, top_k: int) -> pd.DataFrame:
        cols = [config.ID_COL, "fonte", config.TEXT_COL, "dificuldade_efetiva", "score"]
        out = df.sort_values("score", ascending=False).head(top_k)[cols].reset_index(names="indice")
        out["conceitos"] = [
            ", ".join(sorted(self.conceitos[i])) for i in out["indice"]
        ]
        return out

    # ------------------------------------------------------------------ APIs
    def recomendar_por_texto(
        self,
        texto: str,
        conceitos: set[str] | None = None,
        nivel_alvo: str | None = None,
        top_k: int = 5,
        excluir_idx: list[int] | None = None,
        janela_nivel: int = 1,
        dup_threshold: float | None = 0.95,
    ) -> pd.DataFrame:
        """Recomenda questões do catálogo semelhantes a um enunciado.

        Similaridade por conceitos (principal) + TF-IDF; filtra por dificuldade
        dentro de ±``janela_nivel`` do ``nivel_alvo`` (a dificuldade prevista da
        questão). É o caminho usado para questões novas (uso-fim da ferramenta).

        ``dup_threshold``: descarta candidatas com cosseno TF-IDF acima desse
        valor (a MESMA questão, quase idêntica) — útil quando a base tem
        duplicatas entre fontes. Use ``None`` para manter as quase-idênticas.
        """
        vetor = self.vectorizer.transform([data_utils.clean_text(str(texto))])
        cos = self._cosseno(vetor)
        df = self.df.copy()
        df["score"] = self._score_vec(cos, conceitos)

        mask = pd.Series(True, index=df.index)
        if excluir_idx:
            mask &= ~df.index.isin(excluir_idx)
        if dup_threshold is not None:
            mask &= cos < dup_threshold
        aceitos = self._niveis_aceitos(nivel_alvo, janela_nivel)
        if aceitos is not None:
            mask &= df["dificuldade_efetiva"].map(_norm_nivel).isin(aceitos)
        return self._saida(df[mask], top_k)

    def recomendar(self, resolvidos_idx, nivel_alvo=None, top_k: int = 5, janela_nivel: int = 1) -> pd.DataFrame:
        """Recomenda a partir do histórico do aluno (centroide das resolvidas)."""
        df = self.df.copy()
        if resolvidos_idx:
            perfil = np.asarray(self.X[resolvidos_idx].mean(axis=0)).reshape(1, -1)
            conceitos_perfil: set[str] = set().union(*[self.conceitos[i] for i in resolvidos_idx]) \
                if self.tem_conceitos else set()
            df["score"] = self._score_vec(self._cosseno(perfil), conceitos_perfil or None)
        else:
            df["score"] = 0.0

        mask = ~df.index.isin(resolvidos_idx)
        aceitos = self._niveis_aceitos(nivel_alvo, janela_nivel)
        if aceitos is not None:
            mask &= df["dificuldade_efetiva"].map(_norm_nivel).isin(aceitos)
        return self._saida(df[mask], top_k)

    def recomendar_proximo_nivel(self, resolvidos_idx, top_k: int = 5) -> pd.DataFrame:
        """Recomenda no nível do que o aluno mais resolveu (e adjacentes)."""
        if not resolvidos_idx:
            return self.recomendar(resolvidos_idx, top_k=top_k)
        niveis = self.df.loc[resolvidos_idx, "dificuldade_efetiva"].map(_norm_nivel)
        niveis = niveis[niveis.isin(self.ordem)]
        nivel_atual = niveis.mode().iloc[0] if len(niveis) else self.ordem[0]
        print(f"Nível predominante do aluno: {nivel_atual}")
        return self.recomendar(resolvidos_idx, nivel_alvo=nivel_atual, top_k=top_k)


def _resumo_enunciado(texto: str, n: int = 70) -> str:
    txt = " ".join(str(texto).split())
    return txt[:n] + ("…" if len(txt) > n else "")


def run(top_k: int = 5, niveis: int = 3) -> None:
    rec = RecomendadorConteudo(niveis=niveis)
    print(f"Fontes: {', '.join(rec.fontes)} | níveis: {niveis} | "
          f"conceitos: {'sim' if rec.tem_conceitos else 'não (fallback TF-IDF)'}\n")

    rotuladas = rec.df[rec.df["dificuldade_efetiva"].map(_norm_nivel) == rec.ordem[0]]
    if len(rotuladas):
        resolvidos = rotuladas.sample(min(3, len(rotuladas)), random_state=config.RANDOM_SEED).index.tolist()
    else:
        resolvidos = rec.df.sample(min(3, len(rec.df)), random_state=config.RANDOM_SEED).index.tolist()
    print(f"Aluno (demo) resolveu os índices: {resolvidos}\n")

    print(f"=== Recomendações (nível do aluno e adjacentes, top {top_k}) ===")
    df = rec.recomendar_proximo_nivel(resolvidos, top_k=top_k)
    for _, row in df.iterrows():
        print(f"\n{'=' * 80}")
        print(f"[{row['fonte']} #{row[config.ID_COL]}] "
              f"dificuldade={row['dificuldade_efetiva']} | score={row['score']:.3f}")
        if row.get("conceitos"):
            print(f"conceitos: {row['conceitos']}")
        print(f"{'-' * 80}")
        print(row[config.TEXT_COL])


def main() -> None:
    parser = argparse.ArgumentParser(description="Recomendador de exercícios (demo)")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--niveis", type=int, choices=[5, 3], default=3,
                        help="granularidade da dificuldade (padrão 3)")
    args = parser.parse_args()
    run(top_k=args.top_k, niveis=args.niveis)


if __name__ == "__main__":
    main()
