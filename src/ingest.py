"""Etapa 1 — Ingestão: monta a base a partir da pasta ``arquivos/``.

Os dados entregues vêm separados:

  - ``arquivos/txt/exercise_<id>.txt``           -> enunciado puro
  - ``arquivos/txt_with_example/exercise_<id>.txt`` -> enunciado + casos de exemplo
  - ``arquivos/feedbacks_<ano>.json``            -> avaliações dos alunos (nota 1-5)

Este módulo casa enunciados e avaliações pelo ``id`` (extraído do nome do
arquivo), agrega a dificuldade de cada questão a partir das notas dos alunos e
gera ``data/raw/questoes.csv`` com **uma linha por questão**, no formato que o
restante do pipeline (preprocess, train_ml, ...) espera.

Regra de rótulo (5 níveis, casando direto com a nota 1-5 dos alunos):
  1 -> muito_facil | 2 -> facil | 3 -> medio | 4 -> dificil | 5 -> muito_dificil

Por padrão (LABEL_STRATEGY="media") a nota da questão é a média aritmética das
avaliações arredondada ao inteiro mais próximo. Isso evita classificações falsas
em casos bimodais (ex.: metade vota 1 e metade vota 5 -> média ~3 -> medio).

Questões sem nenhuma avaliação ficam com rótulo vazio: entram na base (úteis
para o recomendador), mas são ignoradas pelas etapas supervisionadas.

Uso:
    python -m src.ingest
    python -m src.ingest --sem-exemplos --estrategia moda
"""
from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config

_RE_ID = re.compile(r"exercise_(\d+)", re.IGNORECASE)


# ----------------------------------------------------------------------------
# Leitura dos enunciados
# ----------------------------------------------------------------------------
def _id_do_arquivo(nome: str) -> int | None:
    m = _RE_ID.search(nome)
    return int(m.group(1)) if m else None


def carregar_enunciados(usar_exemplos: bool = True) -> dict[int, str]:
    """Lê os enunciados (.txt) e retorna {id: texto}.

    usar_exemplos=True usa a versão com casos de teste de exemplo
    (txt_with_example); caso contrário, o enunciado puro (txt).
    """
    pasta = config.TXT_EXEMPLOS_DIR if usar_exemplos else config.TXT_DIR
    if not pasta.is_dir():
        raise FileNotFoundError(
            f"Pasta de enunciados não encontrada: {pasta}. "
            "Confirme que os arquivos foram extraídos em arquivos/."
        )
    enunciados: dict[int, str] = {}
    for arq in sorted(pasta.glob("exercise_*.txt")):
        if arq.name.startswith("._"):   # lixo de compactação do macOS
            continue
        qid = _id_do_arquivo(arq.name)
        if qid is None:
            continue
        enunciados[qid] = arq.read_text(encoding="utf-8", errors="ignore").strip()
    print(f"Enunciados lidos de {pasta.name}/: {len(enunciados)}")
    return enunciados


# ----------------------------------------------------------------------------
# Leitura das avaliações dos alunos
# ----------------------------------------------------------------------------
def carregar_feedbacks() -> pd.DataFrame:
    """Lê todos os ``arquivos/feedbacks_*.json`` num único DataFrame."""
    arquivos = sorted(config.ARQUIVOS_DIR.glob("feedbacks_*.json"))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum feedbacks_*.json encontrado em {config.ARQUIVOS_DIR}."
        )
    partes: list[pd.DataFrame] = []
    for arq in arquivos:
        with arq.open(encoding="utf-8") as fh:
            registros = json.load(fh)
        df = pd.DataFrame(registros)
        df["fonte"] = arq.stem   # ex.: feedbacks_2025
        partes.append(df)
        print(f"  {arq.name}: {len(df)} avaliações")
    return pd.concat(partes, ignore_index=True)


# ----------------------------------------------------------------------------
# Agregação das notas em rótulo de dificuldade
# ----------------------------------------------------------------------------
def _faixa(nota: int) -> str:
    """Mapeia uma nota inteira 1-5 para o rótulo canônico de 5 níveis."""
    nota = max(1, min(5, int(nota)))
    return config.NOTA_PARA_ROTULO[nota]


def _arredondar(x: float) -> int:
    """Arredonda ao inteiro mais próximo (meio para cima: 2.5 -> 3)."""
    return int(math.floor(x + 0.5))


def _moda(notas: list[int]) -> int:
    """Nota mais frequente; em caso de empate, a mais próxima da média."""
    cont = Counter(notas)
    maxf = max(cont.values())
    candidatos = [n for n, f in cont.items() if f == maxf]
    if len(candidatos) == 1:
        return candidatos[0]
    media = sum(notas) / len(notas)
    return min(candidatos, key=lambda c: (abs(c - media), c))


def _classificar(media: float, moda: int, estrategia: str) -> str:
    if estrategia == "media":
        return _faixa(_arredondar(media))
    if estrategia == "moda":
        return _faixa(moda)
    raise ValueError(
        f"Estratégia de rótulo desconhecida: {estrategia!r}. Use 'media' ou 'moda'."
    )


def agregar_dificuldade(feedbacks: pd.DataFrame, estrategia: str) -> pd.DataFrame:
    """Agrega as avaliações por questão, gerando o rótulo de dificuldade.

    Retorna colunas: id, n_avaliacoes, media_dificuldade, moda_dificuldade,
    dificuldade.
    """
    linhas: list[dict] = []
    for qid, grupo in feedbacks.groupby("question_id"):
        notas = grupo["difficultylevel"].astype(int).tolist()
        media = sum(notas) / len(notas)
        moda = _moda(notas)
        linhas.append({
            "id": int(qid),
            "n_avaliacoes": len(notas),
            "media_dificuldade": round(media, 3),
            "moda_dificuldade": moda,
            "dificuldade": _classificar(media, moda, estrategia),
        })
    return pd.DataFrame(linhas)


# ----------------------------------------------------------------------------
# Montagem final da base
# ----------------------------------------------------------------------------
def build_dataset(
    usar_exemplos: bool | None = None,
    estrategia: str | None = None,
    salvar: bool = True,
) -> pd.DataFrame:
    """Monta a base completa (uma linha por questão) e salva data/raw/questoes.csv."""
    usar_exemplos = config.USE_EXAMPLES if usar_exemplos is None else usar_exemplos
    estrategia = config.LABEL_STRATEGY if estrategia is None else estrategia

    enunciados = carregar_enunciados(usar_exemplos)
    print("Lendo avaliações dos alunos:")
    feedbacks = carregar_feedbacks()
    rotulos = agregar_dificuldade(feedbacks, estrategia)

    # Base com TODAS as questões que têm enunciado; o rótulo é anexado quando há
    # avaliações (left join), ficando vazio para as demais.
    base = pd.DataFrame(
        sorted(enunciados.items()), columns=["id", "enunciado"]
    )
    df = base.merge(rotulos, on="id", how="left")
    df["n_avaliacoes"] = df["n_avaliacoes"].fillna(0).astype(int)

    rotuladas = df["dificuldade"].notna().sum()
    print(f"\nQuestões com enunciado : {len(df)}")
    print(f"Questões com avaliação : {rotuladas}  (rotuladas)")
    print(f"Sem avaliação          : {len(df) - rotuladas}  (só no recomendador)")
    print(f"\nEstratégia de rótulo   : {estrategia}")
    print("Distribuição das classes:")
    print(df["dificuldade"].value_counts(dropna=True).to_string())

    if salvar:
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.QUESTOES_CSV, index=False)
        print(f"\nBase salva em: {config.QUESTOES_CSV}")
    return df


def ensure_dataset() -> Path:
    """Garante que data/raw/questoes.csv exista (gera se faltar) e retorna o caminho."""
    if not config.QUESTOES_CSV.exists():
        print("questoes.csv não encontrado — gerando a partir de arquivos/ ...\n")
        build_dataset()
    return config.QUESTOES_CSV


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 1 - Ingestão dos arquivos do professor")
    parser.add_argument(
        "--sem-exemplos", action="store_true",
        help="usa o enunciado puro (txt) em vez da versão com casos de exemplo",
    )
    parser.add_argument(
        "--estrategia", choices=["media", "moda"], default=None,
        help=f"como agregar as notas no rótulo (padrão: {config.LABEL_STRATEGY})",
    )
    args = parser.parse_args()
    build_dataset(
        usar_exemplos=False if args.sem_exemplos else None,
        estrategia=args.estrategia,
    )


if __name__ == "__main__":
    main()
