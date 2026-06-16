"""Monta data/raw/dataset.csv a partir dos arquivos brutos.

Lê:
  - data/raw/feedbacks_2024.json
  - data/raw/feedbacks_2025.json
  - data/raw/feedbacks_2026.json
  - data/raw/exercises/exercise_n.txt

Gera:
  - data/raw/dataset.csv com colunas: id, enunciado, dificuldade

Uso:
    python -m src.build_dataset
    python -m src.build_dataset --limiar-facil 2.5 --limiar-dificil 3.5
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from . import config

LIMIAR_FACIL: float = 2.5
LIMIAR_DIFICIL: float = 3.5

FEEDBACK_FILES = [
    "feedbacks_2024.json",
    "feedbacks_2025.json",
    "feedbacks_2026.json",
]


def _label(media: float, limiar_facil: float, limiar_dificil: float) -> str:
    if media <= limiar_facil:
        return "facil"
    if media >= limiar_dificil:
        return "dificil"
    return "medio"


def _load_avaliacoes(raw_dir: Path) -> pd.DataFrame:
    """Lê os três JSONs de feedback e agrega a média por questão."""
    frames = []
    for filename in FEEDBACK_FILES:
        path = raw_dir / filename
        if not path.exists():
            print(f"  [aviso] {filename} não encontrado — pulado")
            continue
        with open(path, encoding="utf-8") as f:
            dados = json.load(f)
        df = pd.DataFrame(dados)
        df["_fonte"] = filename
        frames.append(df)
        print(f"  {filename}: {len(df)} avaliações")

    if not frames:
        raise FileNotFoundError(
            f"Nenhum arquivo de feedback encontrado em {raw_dir}"
        )

    df = pd.concat(frames, ignore_index=True)
    df = df.rename(columns={"question_id": "id", "difficultylevel": "nota"})
    df = df[["id", "nota"]].dropna()
    df["nota"] = pd.to_numeric(df["nota"], errors="coerce")
    df = df.dropna(subset=["nota"])

    agg = (
        df.groupby("id")
        .agg(media=("nota", "mean"), n_avaliacoes=("nota", "count"))
        .reset_index()
    )
    return agg


def _load_enunciados(exercises_dir: Path) -> pd.DataFrame:
    """Lê todos os exercise_n.txt e retorna DataFrame com id e enunciado."""
    rows = []
    for txt in exercises_dir.glob("exercise_*.txt"):
        match = re.match(r"exercise_(\d+)\.txt", txt.name)
        if not match:
            continue
        question_id = int(match.group(1))
        enunciado = txt.read_text(encoding="utf-8").strip()
        if enunciado:
            rows.append({"id": question_id, "enunciado": enunciado})

    if not rows:
        raise FileNotFoundError(
            f"Nenhum arquivo exercise_*.txt encontrado em {exercises_dir}"
        )
    return pd.DataFrame(rows)


def run(
    limiar_facil: float = LIMIAR_FACIL,
    limiar_dificil: float = LIMIAR_DIFICIL,
) -> pd.DataFrame:
    raw_dir = config.RAW_DIR
    exercises_dir = raw_dir / "exercises"

    if not exercises_dir.exists():
        raise FileNotFoundError(f"Pasta não encontrada: {exercises_dir}")

    print("Carregando avaliações...")
    df_aval = _load_avaliacoes(raw_dir)
    print(f"  {len(df_aval)} questões com avaliação | "
          f"{df_aval['n_avaliacoes'].sum():.0f} avaliações no total")

    print("\nCarregando enunciados...")
    df_enun = _load_enunciados(exercises_dir)
    print(f"  {len(df_enun)} arquivos .txt lidos")

    # Inner join: só questões com enunciado E avaliação
    df = df_enun.merge(df_aval, on="id", how="inner")
    sem_avaliacao = len(df_enun) - len(df)
    print(f"  {sem_avaliacao} questão(ões) sem avaliação — ignoradas")

    # Aplica os limiares
    df["dificuldade"] = df["media"].apply(
        lambda m: _label(m, limiar_facil, limiar_dificil)
    )

    # Salva
    saida = raw_dir / "dataset.csv"
    df[["id", "enunciado", "dificuldade"]].to_csv(saida, index=False)

    print(f"\n=== Dataset gerado: {saida} ===")
    print(f"Total de questões : {len(df)}")
    print(f"Limiares          : facil <= {limiar_facil} | dificil >= {limiar_dificil}")
    print("\nDistribuição das classes:")
    print(df["dificuldade"].value_counts())
    print("\nEstatísticas da média de dificuldade:")
    print(df["media"].describe().round(2))

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Monta data/raw/dataset.csv")
    parser.add_argument("--limiar-facil", type=float, default=LIMIAR_FACIL)
    parser.add_argument("--limiar-dificil", type=float, default=LIMIAR_DIFICIL)
    args = parser.parse_args()
    run(limiar_facil=args.limiar_facil, limiar_dificil=args.limiar_dificil)


if __name__ == "__main__":
    main()