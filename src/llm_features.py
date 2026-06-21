"""Etapa 5 — LLM como extrator de características.

Pede ao LLM (Groq/Llama) para identificar quais conceitos/algoritmos aparecem em
cada enunciado (recursão, programação dinâmica, grafos, etc.) e gera colunas
binárias que podem ser combinadas com as features TF-IDF (ver src.evaluate).

Tem retomada automática: linhas já processadas em data/processed/llm_features.csv
são puladas, permitindo rodar aos poucos sem refazer chamadas.

Pré-requisito: GROQ_API_KEY no .env e dataset_limpo.csv.

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

from . import config, llm_client

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


def _vetor_conceitos(brutos) -> dict[str, int]:
    """Converte uma lista de conceitos crus em {conceito: 0/1} na taxonomia."""
    encontrados = {_norm(c) for c in brutos} if isinstance(brutos, list) else set()
    return {c: int(c in encontrados) for c in CONCEITOS}


def extract_one(statement: str) -> dict[str, int]:
    """Retorna {conceito: 0/1} para um enunciado."""
    data = llm_client.generate_json(_build_prompt(statement), SYSTEM)
    brutos = data.get("conceitos", []) if isinstance(data, dict) else data
    return _vetor_conceitos(brutos)


def _build_prompt_lote(statements: list[str], max_chars: int = 1500) -> str:
    """Monta um prompt que extrai conceitos de VÁRIOS enunciados de uma vez."""
    lista = ", ".join(CONCEITOS)
    partes = [
        "Para CADA enunciado abaixo, identifique os conceitos/algoritmos "
        f"necessários para resolvê-lo, usando APENAS nomes desta lista: {lista}.",
        "Responda APENAS em JSON: um array com um objeto por enunciado, no "
        'formato [{"id": <numero>, "conceitos": ["nome1", "nome2"]}]. '
        "Inclua TODOS os ids e nada fora do JSON.",
    ]
    for i, texto in enumerate(statements, 1):
        partes.append(f"\nEnunciado {i}:\n{str(texto)[:max_chars]}")
    return "\n".join(partes)


def extract_batch(statements: list[str]) -> list[dict[str, int]]:
    """Extrai conceitos de vários enunciados em UMA chamada (economiza cota).

    Casa a resposta por ``id``; itens não cobertos são reprocessados
    individualmente, preservando tamanho e corretude do resultado.
    """
    data = llm_client.generate_json(_build_prompt_lote(statements), SYSTEM)
    por_id: dict[int, dict[str, int]] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "id" in item:
                try:
                    idx = int(item["id"])
                except (TypeError, ValueError):
                    continue
                por_id[idx] = _vetor_conceitos(item.get("conceitos", []))
    return [
        por_id.get(i) or extract_one(str(statements[i - 1]))
        for i in range(1, len(statements) + 1)
    ]


def _load_cache() -> pd.DataFrame:
    if config.LLM_FEATURES.exists():
        return pd.read_csv(config.LLM_FEATURES)
    return pd.DataFrame(columns=[ROW_KEY, *CONCEITOS])


def run(n: int | None = None, sleep: float = 0.0, lote: int = 1) -> None:
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

    print(f"Extraindo conceitos de {len(pendentes)} enunciado(s) via LLM (Groq)...")
    if lote > 1:
        print(f"Lote (prompt packing): {lote} enunciados por requisição.")
    registros = pendentes.to_dict("records")
    novas = []
    if lote and lote > 1:
        grupos = [registros[i:i + lote] for i in range(0, len(registros), lote)]
        for grupo in tqdm(grupos, total=len(grupos), desc=f"LLM (features, lote={lote})"):
            vetores = extract_batch([str(r[config.TEXT_COL]) for r in grupo])
            for r, feats in zip(grupo, vetores):
                feats = dict(feats)
                feats[ROW_KEY] = int(r[ROW_KEY])
                novas.append(feats)
            if sleep:
                time.sleep(sleep)
    else:
        for r in tqdm(registros, total=len(registros), desc="LLM (features)"):
            feats = extract_one(str(r[config.TEXT_COL]))
            feats[ROW_KEY] = int(r[ROW_KEY])
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
    parser = argparse.ArgumentParser(description="Etapa 5 - Extração de conceitos com LLM (Groq)")
    parser.add_argument("--n", type=int, default=None, help="limita qtd. de linhas pendentes")
    parser.add_argument("--sleep", type=float, default=0.0, help="pausa entre chamadas (s)")
    parser.add_argument("--lote", type=int, default=1,
                        help="enunciados por requisição (prompt packing; >1 economiza cota)")
    args = parser.parse_args()
    run(n=args.n, sleep=args.sleep, lote=args.lote)


if __name__ == "__main__":
    main()
