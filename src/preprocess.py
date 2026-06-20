"""Etapa 2 — Pré-processamento: limpeza de texto + vetorização TF-IDF.

Lê a base bruta de data/raw/, limpa os enunciados, ajusta um TF-IDF e salva:
  - data/processed/dataset_limpo.csv  (id, enunciado, rótulo, texto_limpo)
  - models/tfidf_vectorizer.joblib    (vetorizador ajustado)
  - models/tfidf_matrix.joblib        (matriz esparsa X)

O ``dataset_limpo.csv`` é o insumo central das etapas seguintes (treino e
avaliação). O vetorizador/matriz salvos aqui são ajustados sobre TODA a base e
servem para EDA e para o recomendador; a avaliação supervisionada (``train_ml``,
``evaluate``) NÃO os reutiliza — lá o TF-IDF é reajustado só com os dados de
treino de cada fold, para não vazar informação do teste.

Uso:
    python -m src.preprocess
    python -m src.preprocess --max-features 8000 --ngram 2
"""
from __future__ import annotations

import argparse

import joblib

from . import config, data_utils, ingest


def run(max_features: int | None = None, ngram_max: int | None = None,
        min_df: int | None = None) -> None:
    # Etapa 1: garante que data/raw/questoes.csv exista (gera de arquivos/ se faltar).
    ingest.ensure_dataset()

    df = data_utils.load_raw()
    data_utils.check_columns(df, [config.TEXT_COL, config.LABEL_COL])

    # Remove apenas linhas sem enunciado; as sem rótulo (questões sem avaliação)
    # são mantidas para o recomendador, mas padronizamos o rótulo quando existe.
    df = df.dropna(subset=[config.TEXT_COL]).reset_index(drop=True)
    rotulo = df[config.LABEL_COL].astype("string").str.strip().str.lower()
    df[config.LABEL_COL] = rotulo.where(rotulo.notna() & (rotulo != "nan"))

    print("Limpando enunciados...")
    df["texto_limpo"] = df[config.TEXT_COL].astype(str).map(data_utils.clean_text)

    # Descarta enunciados que ficaram vazios após a limpeza.
    df = df[df["texto_limpo"].str.len() > 0].reset_index(drop=True)

    vectorizer = data_utils.build_vectorizer(max_features, ngram_max, min_df)
    print(
        f"Ajustando TF-IDF (max_features={vectorizer.max_features}, "
        f"ngram={vectorizer.ngram_range}, min_df={vectorizer.min_df})..."
    )
    X = vectorizer.fit_transform(df["texto_limpo"])

    # Seleciona colunas úteis para salvar (id é opcional).
    cols = [c for c in (config.ID_COL, config.TEXT_COL, config.LABEL_COL) if c in df.columns]
    cols.append("texto_limpo")
    saida = data_utils.save_processed(df[cols], config.CLEAN_DATASET.name)

    joblib.dump(vectorizer, config.TFIDF_VECTORIZER)
    joblib.dump(X, config.TFIDF_MATRIX)

    print("\n=== Pré-processamento concluído ===")
    print(f"Exemplos válidos : {X.shape[0]}")
    print(f"Features TF-IDF  : {X.shape[1]}")
    print(f"Dataset limpo    : {saida}")
    print(f"Vetorizador      : {config.TFIDF_VECTORIZER}")
    print("\nDistribuição das classes:")
    print(df[config.LABEL_COL].value_counts())


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 2 - Pré-processamento e TF-IDF")
    parser.add_argument("--max-features", type=int, default=None,
                        help=f"padrão: config.TFIDF_MAX_FEATURES ({config.TFIDF_MAX_FEATURES})")
    parser.add_argument("--ngram", type=int, default=None,
                        help=f"n-grama máximo (1..n); padrão: {config.TFIDF_NGRAM_MAX}")
    parser.add_argument("--min-df", type=int, default=None,
                        help=f"padrão: config.TFIDF_MIN_DF ({config.TFIDF_MIN_DF})")
    args = parser.parse_args()
    run(max_features=args.max_features, ngram_max=args.ngram, min_df=args.min_df)


if __name__ == "__main__":
    main()
