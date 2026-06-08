"""Etapa 5 — LLM como extrator de características.

Pede ao Gemini para identificar quais conceitos/algoritmos aparecem em cada
enunciado (recursão, programação dinâmica, grafos, etc.) e gera colunas
binárias que podem ser combinadas com as features TF-IDF (ver src.evaluate).

Tem retomada automática: linhas já processadas em data/processed/llm_features.csv
são puladas, permitindo rodar aos poucos sem refazer chamadas.

Pré-requisito: GEMINI_API_KEY no .env e dataset_limpo.csv.

Uso:
    python -m src.llm_features            # processa toda a base
    python -m src.llm_features --n 50     # processa só as 50 primeiras pendentes
"""
from __future__ import annotations

import argparse
import time
import unicodedata

import pandas as pd
from tqdm import tqdm

from . import config, gemini_client

# Taxonomia de conceitos considerada (ajuste conforme o domínio da base).
CONCEITOS = [
    "matematica",
    "implementacao",
    "strings",
    "estruturas_de_dados",
    "recursao",
    "programacao_dinamica",
    "grafos",
    "arvores",
    "busca_binaria",
    "guloso",
    "ordenacao",
    "geometria",
    "teoria_dos_numeros",
    "backtracking",
    "forca_bruta",
]

SYSTEM = (
    "Você é um especialista em algoritmos e maratonas de programação. "
    "Identifique os conceitos necessários para resolver cada questão."
)

ROW_KEY = "_row"   # chave de junção com o dataset_limpo.csv


def _norm(token: str) -> str:
    txt = unicodedata.normalize("NFKD", str(token).lower().strip())
    txt = "".join(c for c in txt if not unicodedata.combining(c))
    return txt.replace(" ", "_").replace("-", "_")


def _build_prompt(statement: str) -> str:
    lista = ", ".join(CONCEITOS)
    return (
        "Analise o enunciado e identifique quais conceitos/algoritmos são "
        "necessários para resolvê-lo. Use APENAS nomes desta lista: "
        f"{lista}.\n"
        'Responda em JSON no formato {"conceitos": ["nome1", "nome2"]}.\n\n'
        f"Enunciado:\n{statement[:4000]}"
    )


def extract_one(statement: str) -> dict[str, int]:
    """Retorna {conceito: 0/1} para um enunciado."""
    data = gemini_client.generate_json(_build_prompt(statement), SYSTEM)
    brutos = data.get("conceitos", []) if isinstance(data, dict) else data
    encontrados = {_norm(c) for c in brutos} if isinstance(brutos, list) else set()
    return {c: int(c in encontrados) for c in CONCEITOS}


def _load_cache() -> pd.DataFrame:
    if config.LLM_FEATURES.exists():
        return pd.read_csv(config.LLM_FEATURES)
    return pd.DataFrame(columns=[ROW_KEY, *CONCEITOS])


def run(n: int | None = None, sleep: float = 0.0) -> None:
    config.require_api_key()
    if not config.CLEAN_DATASET.exists():
        raise FileNotFoundError(
            "dataset_limpo.csv não encontrado. Rode antes: python -m src.preprocess"
        )

    df = pd.read_csv(config.CLEAN_DATASET).reset_index(drop=True)
    df[ROW_KEY] = df.index

    cache = _load_cache()
    feitos = set(cache[ROW_KEY].tolist()) if len(cache) else set()
    pendentes = df[~df[ROW_KEY].isin(feitos)]
    if n is not None:
        pendentes = pendentes.head(n)

    if pendentes.empty:
        print("Nada a processar — todas as linhas já estão no cache.")
        return

    print(f"Extraindo conceitos de {len(pendentes)} enunciado(s) via Gemini...")
    novas = []
    for _, row in tqdm(pendentes.iterrows(), total=len(pendentes), desc="Gemini (features)"):
        feats = extract_one(str(row[config.TEXT_COL]))
        feats[ROW_KEY] = int(row[ROW_KEY])
        novas.append(feats)
        if sleep:
            time.sleep(sleep)

    resultado = pd.concat([cache, pd.DataFrame(novas)], ignore_index=True)
    resultado = resultado.sort_values(ROW_KEY)[[ROW_KEY, *CONCEITOS]]
    resultado.to_csv(config.LLM_FEATURES, index=False)

    print(f"\nFeatures salvas em: {config.LLM_FEATURES} ({len(resultado)} linhas)")
    print("Frequência de cada conceito:")
    print(resultado[CONCEITOS].sum().sort_values(ascending=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 5 - Extração de conceitos com Gemini")
    parser.add_argument("--n", type=int, default=None, help="limita qtd. de linhas pendentes")
    parser.add_argument("--sleep", type=float, default=0.0, help="pausa entre chamadas (s)")
    args = parser.parse_args()
    run(n=args.n, sleep=args.sleep)


if __name__ == "__main__":
    main()
