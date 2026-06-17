"""Etapa 2 — Pré-processamento: limpeza de texto + vetorização TF-IDF.

Lê a base bruta de data/raw/, limpa os enunciados, ajusta um TF-IDF e salva:
  - data/processed/dataset_limpo.csv  (id, enunciado, rótulo, texto_limpo)
  - models/tfidf_vectorizer.joblib    (vetorizador ajustado)
  - models/tfidf_matrix.joblib        (matriz esparsa X)

Uso:
    python -m src.preprocess
    python -m src.preprocess --max-features 8000 --ngram 2
"""
from __future__ import annotations

import argparse

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config, data_utils, ingest


def run(max_features: int = 5000, ngram_max: int = 2, min_df: int = 2) -> None:
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

    print(f"Ajustando TF-IDF (max_features={max_features}, ngram=(1,{ngram_max}))...")
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, ngram_max),
        min_df=min_df,
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
    parser.add_argument("--max-features", type=int, default=5000)
    parser.add_argument("--ngram", type=int, default=2, help="n-grama máximo (1..n)")
    parser.add_argument("--min-df", type=int, default=2)
    args = parser.parse_args()
    run(max_features=args.max_features, ngram_max=args.ngram, min_df=args.min_df)


if __name__ == "__main__":
    main()
