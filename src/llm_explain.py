"""Etapa 6 — LLM como explicador.

Gera justificativas em linguagem natural para a dificuldade atribuída a um
enunciado (seja a classificação do modelo de ML, seja a do próprio LLM).
Útil para dar transparência às predições e apoiar o aluno/professor.

Pré-requisito: GROQ_API_KEY no .env.

Uso:
    python -m src.llm_explain --n 5
"""
from __future__ import annotations

import argparse

import pandas as pd

from . import config, data_utils, llm_client

SYSTEM = (
    "Você é um professor de algoritmos. Explique de forma didática e objetiva "
    "por que uma questão de programação tem determinada dificuldade."
)


def _build_prompt(statement: str, dificuldade: str) -> str:
    return (
        f"Uma questão foi classificada como de dificuldade '{dificuldade}'.\n"
        "Explique em 2 a 4 frases o porquê, citando os conceitos/algoritmos "
        "envolvidos e o que torna a questão mais fácil ou mais difícil.\n\n"
        f"Enunciado:\n{statement[:4000]}"
    )


def explain(statement: str, dificuldade: str) -> str:
    """Retorna uma explicação textual para a dificuldade informada."""
    return llm_client.generate(
        _build_prompt(statement, dificuldade), SYSTEM, temperature=0.3
    )


def run(n: int = 5) -> None:
    config.require_api_key()
    if not config.CLEAN_DATASET.exists():
        raise FileNotFoundError(
            "dataset_limpo.csv não encontrado. Rode antes: python -m src.preprocess"
        )

    df = pd.read_csv(config.CLEAN_DATASET)
    _, df_test = data_utils.split_train_test(df)
    amostra = df_test.sample(min(n, len(df_test)), random_state=config.RANDOM_SEED)

    for _, row in amostra.iterrows():
        rotulo = str(row.get(config.LABEL_COL, "desconhecida"))
        texto = str(row[config.TEXT_COL])
        print("=" * 70)
        print(f"Dificuldade: {rotulo}")
        print(f"Enunciado: {texto[:300]}{'...' if len(texto) > 300 else ''}")
        print("-> Explicação do LLM:")
        print(explain(texto, rotulo))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 6 - Explicações com LLM (Groq)")
    parser.add_argument("--n", type=int, default=5, help="qtd. de exemplos a explicar")
    args = parser.parse_args()
    run(n=args.n)


if __name__ == "__main__":
    main()
