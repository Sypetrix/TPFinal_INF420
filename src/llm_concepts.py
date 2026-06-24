"""Extração de conceitos via LLM (função central do pipeline).

Pede ao LLM (provedor configurável, padrão Groq/Llama) para identificar quais
conceitos/algoritmos aparecem em cada enunciado (recursão, programação dinâmica,
grafos, etc.). A saída estruturada (JSON) é usada em dois lugares:

  * como **feature adicional** dos modelos de ML (junto ao TF-IDF) — base de
    treino, cacheada em data/processed/<DATASET>/llm_features.csv (ver evaluate);
  * como **critério de similaridade** do recomendador — base de recomendação,
    cacheada por fonte em data/processed/<fonte>/conceitos.csv.

Tem retomada automática (não refaz itens já no cache).

Pré-requisito: chave do provedor de LLM no .env (LLM_PROVIDER) e, no modo base de
treino, dataset_limpo.csv.

Uso:
    python -m src.llm_concepts                 # base de treino (toda)
    python -m src.llm_concepts --n 50          # base de treino (50 pendentes)
    python -m src.llm_concepts --fonte SPOJ --lote 10   # base de recomendação (1 fonte)
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


def _salvar_cache_atomico(cache: pd.DataFrame, path) -> None:
    """Grava o cache CSV de forma atômica (escreve em .tmp e renomeia).

    Usado para salvar a CADA lote, de modo que um crash (cota, JSON quebrado,
    queda de rede) nunca perca o trabalho já feito.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    cache.to_csv(tmp, index=False)
    tmp.replace(path)


# ----------------------------------------------------------------------------
# Conceitos para a BASE DE RECOMENDAÇÃO (por fonte, com cache/retomada)
# ----------------------------------------------------------------------------
def _conceitos_csv(fonte: str):
    """Caminho do cache de conceitos de uma fonte do recomendador."""
    return config.DATA_DIR / "processed" / fonte.strip() / "conceitos.csv"


def extrair_conceitos_fonte(fonte: str, n: int | None = None, lote: int = 1,
                            sleep: float = 0.0):
    """Extrai e cacheia os conceitos das questões de uma fonte (recomendador).

    Lê data/raw/<fonte>/questoes.csv e grava data/processed/<fonte>/conceitos.csv
    (colunas: id + um 0/1 por conceito). Tem retomada (não refaz ids já no cache).
    """
    config.require_api_key()
    csv_q = config.questoes_csv_for(fonte)
    if not csv_q.exists():
        raise FileNotFoundError(
            f"Base da fonte '{fonte}' não encontrada em {csv_q}. "
            f"Rode antes: DATASET={fonte} python -m src.ingest"
        )
    df = pd.read_csv(csv_q).dropna(subset=[config.TEXT_COL])
    saida = _conceitos_csv(fonte)
    saida.parent.mkdir(parents=True, exist_ok=True)
    cache = (pd.read_csv(saida) if saida.exists()
             else pd.DataFrame(columns=[config.ID_COL, *CONCEITOS]))
    feitos = set(cache[config.ID_COL].astype(str)) if len(cache) else set()
    pend = df[~df[config.ID_COL].astype(str).isin(feitos)]
    if n is not None:
        pend = pend.head(n)
    if pend.empty:
        print(f"[{fonte}] conceitos já em cache — nada a fazer.")
        return saida

    registros = pend.to_dict("records")
    print(f"[{fonte}] extraindo conceitos de {len(registros)} questão(ões) via LLM...")
    lote_eff = max(1, int(lote or 1))
    grupos = [registros[i:i + lote_eff] for i in range(0, len(registros), lote_eff)]
    desc = f"conceitos {fonte} (lote={lote_eff})" if lote_eff > 1 else f"conceitos {fonte}"
    # Salva DEPOIS DE CADA LOTE — mesma garantia de retomada do passo de treino.
    for grupo in tqdm(grupos, desc=desc):
        if lote_eff > 1:
            vetores = extract_batch([str(r[config.TEXT_COL]) for r in grupo])
        else:
            vetores = [extract_one(str(grupo[0][config.TEXT_COL]))]
        novas_linhas = []
        for r, feats in zip(grupo, vetores):
            linha = dict(feats)
            linha[config.ID_COL] = r[config.ID_COL]
            novas_linhas.append(linha)
        cache = pd.concat([cache, pd.DataFrame(novas_linhas)], ignore_index=True)
        cache_ordenado = cache[[config.ID_COL, *CONCEITOS]]
        _salvar_cache_atomico(cache_ordenado, saida)
        if sleep:
            time.sleep(sleep)

    res = pd.read_csv(saida)
    print(f"[{fonte}] conceitos salvos em {saida} ({len(res)} linhas)")
    return saida


def carregar_conceitos(fontes) -> dict[tuple[str, str], set[str]]:
    """Carrega os conceitos cacheados de várias fontes do recomendador.

    Retorna ``{(fonte, str(id)): {conceitos}}``. Fontes sem cache são ignoradas
    (o recomendador cai para similaridade TF-IDF quando não há conceitos).
    """
    mapa: dict[tuple[str, str], set[str]] = {}
    for fonte in fontes:
        csv = _conceitos_csv(fonte)
        if not csv.exists():
            continue
        df = pd.read_csv(csv)
        for _, row in df.iterrows():
            conjunto = {c for c in CONCEITOS if int(row.get(c, 0) or 0) == 1}
            mapa[(fonte, str(row[config.ID_COL]))] = conjunto
    return mapa


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

    print(f"Extraindo conceitos de {len(pendentes)} enunciado(s) via LLM...")
    if lote > 1:
        print(f"Lote (prompt packing): {lote} enunciados por requisição.")
    registros = pendentes.to_dict("records")
    lote_eff = max(1, int(lote or 1))
    grupos = [registros[i:i + lote_eff] for i in range(0, len(registros), lote_eff)]
    desc = f"LLM (features, lote={lote_eff})" if lote_eff > 1 else "LLM (features)"
    # Salva o cache DEPOIS DE CADA LOTE (atomicamente). Se a API cair, estourar
    # a cota ou devolver JSON ruim, nada já feito é perdido — retoma do CSV.
    for grupo in tqdm(grupos, total=len(grupos), desc=desc):
        if lote_eff > 1:
            vetores = extract_batch([str(r[config.TEXT_COL]) for r in grupo])
        else:
            vetores = [extract_one(str(grupo[0][config.TEXT_COL]))]
        novas_linhas = []
        for r, feats in zip(grupo, vetores):
            linha = dict(feats)
            linha[ROW_KEY] = int(r[ROW_KEY])
            novas_linhas.append(linha)
        cache = pd.concat([cache, pd.DataFrame(novas_linhas)], ignore_index=True)
        cache_ordenado = cache.sort_values(ROW_KEY)[[ROW_KEY, *CONCEITOS]]
        _salvar_cache_atomico(cache_ordenado, config.LLM_FEATURES)
        if sleep:
            time.sleep(sleep)

    resultado = pd.read_csv(config.LLM_FEATURES)
    print(f"\nFeatures salvas em: {config.LLM_FEATURES} ({len(resultado)} linhas)")
    print("Frequência de cada conceito:")
    print(resultado[CONCEITOS].sum().sort_values(ascending=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Extração de conceitos com LLM (provedor configurável)")
    parser.add_argument("--fonte", type=str, default=None,
                        help="extrai conceitos da base de uma fonte do recomendador "
                             "(data/raw/<fonte>/) em vez da base de treino")
    parser.add_argument("--n", type=int, default=None, help="limita qtd. de itens pendentes")
    parser.add_argument("--sleep", type=float, default=0.0, help="pausa entre chamadas (s)")
    parser.add_argument("--lote", type=int, default=10,
                        help="enunciados por requisição (prompt packing; >1 economiza cota). "
                             "Padrão 10 — reduz ~10x as chamadas à API. Use --lote 1 p/ item-a-item.")
    args = parser.parse_args()
    if args.fonte:
        extrair_conceitos_fonte(args.fonte, n=args.n, lote=args.lote, sleep=args.sleep)
    else:
        run(n=args.n, sleep=args.sleep, lote=args.lote)


if __name__ == "__main__":
    main()
