"""Etapa 1 — Ingestão: monta a base a partir da pasta ``arquivos/``.

A fonte ativa é ``arquivos/<DATASET>/`` (``DATASET`` no ``.env``). A Etapa 1
reconhece dois formatos de fonte (detectados automaticamente, ou forçados por
``DATASET_FORMAT``):

1. ``feedbacks`` — enunciados em arquivos ``.txt`` + avaliações dos alunos
   (ex.: **INF110**). O rótulo de dificuldade é **derivado das notas 1-5** que os
   alunos deram em ``feedbacks_<ano>.json``:

     - ``txt/exercise_<id>.txt``              -> enunciado puro
     - ``txt_with_example/exercise_<id>.txt`` -> enunciado + casos de exemplo
     - ``feedbacks_<ano>.json``               -> avaliações dos alunos (nota 1-5)

   Por padrão (``LABEL_STRATEGY="media"``) a nota da questão é a média aritmética
   das avaliações arredondada ao inteiro mais próximo (evita classificações falsas
   em casos bimodais). Questões sem nenhuma avaliação ficam sem rótulo.

2. ``judge_json`` — um único JSON em que cada questão já traz a dificuldade
   **atribuída pelo juiz** (ex.: **Neps Academy**, ``Neps_Academy_complete.json``).
   Não há notas de alunos para agregar: o rótulo do juiz (``metadata.Difficulty``,
   em português) é mapeado direto para a escala canônica via
   ``config.NEPS_DIFFICULTY_MAP``.

Ambos os formatos geram ``data/raw/<DATASET>/questoes.csv`` com **uma linha por
questão** e as mesmas colunas (id, enunciado, n_avaliacoes, media_dificuldade,
moda_dificuldade, dificuldade), no formato que o restante do pipeline espera.

Escala de rótulos (5 níveis):
  1 -> muito_facil | 2 -> facil | 3 -> medio | 4 -> dificil | 5 -> muito_dificil

Questões sem rótulo entram na base (úteis para o recomendador), mas são
ignoradas pelas etapas supervisionadas.

Uso:
    python -m src.ingest
    python -m src.ingest --sem-exemplos --estrategia moda
    DATASET=Neps python -m src.ingest        # processa a base do Neps Academy
"""
from __future__ import annotations

import argparse
import json
import math
import re
import unicodedata
from collections import Counter
from pathlib import Path

import pandas as pd

from . import config

_RE_ID = re.compile(r"exercise_(\d+)", re.IGNORECASE)

# Colunas da base consolidada (mesmas para todos os formatos de fonte).
_COLUNAS = ["id", "enunciado", "n_avaliacoes", "media_dificuldade",
            "moda_dificuldade", "dificuldade"]


def _normalizar(texto: str) -> str:
    """Minúsculas e sem acento — para casar rótulos do juiz com os mapas."""
    nfkd = unicodedata.normalize("NFKD", str(texto).lower().strip())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


# ----------------------------------------------------------------------------
# Formato "feedbacks" (INF110): enunciados em .txt + notas dos alunos
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
            f"Confirme que os arquivos da fonte '{config.DATASET}' estão em "
            f"{config.DATASET_DIR}."
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


def carregar_feedbacks() -> pd.DataFrame:
    """Lê todos os ``arquivos/<DATASET>/feedbacks_*.json`` num único DataFrame."""
    arquivos = sorted(config.DATASET_DIR.glob("feedbacks_*.json"))
    if not arquivos:
        raise FileNotFoundError(
            f"Nenhum feedbacks_*.json encontrado em {config.DATASET_DIR}."
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


def _build_feedbacks(usar_exemplos: bool, estrategia: str, salvar: bool) -> pd.DataFrame:
    """Monta a base no formato 'feedbacks' (enunciados .txt + notas de alunos)."""
    enunciados = carregar_enunciados(usar_exemplos)
    print("Lendo avaliações dos alunos:")
    feedbacks = carregar_feedbacks()
    rotulos = agregar_dificuldade(feedbacks, estrategia)

    # Base com TODAS as questões que têm enunciado; o rótulo é anexado quando há
    # avaliações (left join), ficando vazio para as demais.
    base = pd.DataFrame(sorted(enunciados.items()), columns=["id", "enunciado"])
    df = base.merge(rotulos, on="id", how="left")
    df["n_avaliacoes"] = df["n_avaliacoes"].fillna(0).astype(int)

    print(f"\nEstratégia de rótulo   : {estrategia}")
    return _resumir_e_salvar(df, salvar)


# ----------------------------------------------------------------------------
# Formato "judge_json" (Neps Academy): JSON com dificuldade do juiz
# ----------------------------------------------------------------------------
def _encontrar_json_juiz() -> Path:
    """Acha o JSON de questões da fonte (qualquer .json que não seja feedbacks)."""
    candidatos = [
        p for p in sorted(config.DATASET_DIR.glob("*.json"))
        if not p.name.startswith("feedbacks_") and not p.name.startswith("._")
    ]
    if not candidatos:
        raise FileNotFoundError(
            f"Nenhum JSON de questões encontrado em {config.DATASET_DIR} "
            f"(esperado um arquivo como '<fonte>_complete.json')."
        )
    return candidatos[0]


def _carregar_json_juiz(path: Path) -> list[dict]:
    """Lê o JSON do juiz, tolerando o ``]`` final ausente (arquivo truncado)."""
    raw = path.read_text(encoding="utf-8")
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        texto = raw.rstrip()
        if texto.endswith(","):
            texto = texto[:-1]
        if not texto.endswith("]"):
            texto = texto + "\n]"
            print(f"  [aviso] {path.name} sem ']' final — array fechado em memória "
                  "(arquivo bruto não foi alterado)")
        return json.loads(texto)


def _texto_juiz(rec: dict, usar_exemplos: bool) -> str:
    """Compõe o enunciado de um registro do juiz.

    Sem exemplos: descrição + seções de Entrada/Saída. Com exemplos
    (usar_exemplos=True): acrescenta os casos de teste, espelhando o
    txt_with_example/ do formato 'feedbacks'.
    """
    partes: list[str] = []
    desc = (rec.get("Problem_Description") or "").strip()
    if desc:
        partes.append(desc)
    entrada = (rec.get("Input") or "").strip()
    if entrada:
        partes.append("Entrada\n" + entrada)
    saida = (rec.get("Output") or "").strip()
    if saida:
        partes.append("Saída\n" + saida)
    if usar_exemplos:
        for i, caso in enumerate(rec.get("Test_Case") or [], 1):
            ci = (caso.get("input") or "").strip()
            co = (caso.get("output") or "").strip()
            if ci or co:
                partes.append(f"Exemplo {i}\nEntrada: {ci}\nSaída: {co}")
    return "\n\n".join(partes).strip()


def _build_judge_json(usar_exemplos: bool, salvar: bool) -> pd.DataFrame:
    """Monta a base no formato 'judge_json' (dificuldade já dada pelo juiz)."""
    path = _encontrar_json_juiz()
    print(f"Lendo questões do juiz: {path.name}")
    registros = _carregar_json_juiz(path)
    print(f"  registros no arquivo: {len(registros)}")

    linhas: list[dict] = []
    nao_mapeadas: Counter = Counter()
    for rec in registros:
        qid = rec.get("ID")
        if qid is None:
            continue
        bruta = ((rec.get("metadata") or {}).get("Difficulty") or "").strip()
        rotulo = config.NEPS_DIFFICULTY_MAP.get(_normalizar(bruta)) if bruta else None
        if bruta and rotulo is None:
            nao_mapeadas[bruta] += 1
        nota = config.ROTULO_PARA_NOTA.get(rotulo) if rotulo else None
        linhas.append({
            "id": int(qid),
            "enunciado": _texto_juiz(rec, usar_exemplos),
            # n_avaliacoes/media/moda não se aplicam (rótulo é do juiz, não de
            # alunos); preenchidos com o próprio nível para manter o esquema.
            "n_avaliacoes": 1 if rotulo else 0,
            "media_dificuldade": float(nota) if nota else None,
            "moda_dificuldade": float(nota) if nota else None,
            "dificuldade": rotulo,
        })

    df = (
        pd.DataFrame(linhas, columns=_COLUNAS)
        .drop_duplicates("id")
        .sort_values("id")
        .reset_index(drop=True)
    )

    if nao_mapeadas:
        print("  [aviso] rótulos de dificuldade não reconhecidos (ficam sem rótulo):")
        for rotulo, n in nao_mapeadas.most_common():
            print(f"    {rotulo!r}: {n}")

    print("\nFonte do rótulo        : juiz (metadata.Difficulty)")
    return _resumir_e_salvar(df, salvar)


# ----------------------------------------------------------------------------
# Resumo + salvamento (comum aos dois formatos)
# ----------------------------------------------------------------------------
def _resumir_e_salvar(df: pd.DataFrame, salvar: bool) -> pd.DataFrame:
    """Imprime o resumo padrão e salva data/raw/<DATASET>/questoes.csv."""
    rotuladas = df["dificuldade"].notna().sum()
    print(f"\nQuestões com enunciado : {len(df)}")
    print(f"Questões com rótulo    : {rotuladas}  (rotuladas)")
    print(f"Sem rótulo             : {len(df) - rotuladas}  (só no recomendador)")
    print("\nDistribuição das classes:")
    vc = df["dificuldade"].value_counts(dropna=True)
    vc = vc.reindex([c for c in config.DIFFICULTY_LABELS if c in vc.index])
    print(vc.to_string())

    if salvar:
        config.RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(config.QUESTOES_CSV, index=False)
        print(f"\nBase salva em: {config.QUESTOES_CSV}")
    return df


# ----------------------------------------------------------------------------
# Detecção de formato + montagem final
# ----------------------------------------------------------------------------
def _detectar_formato() -> str:
    """Decide o formato da fonte ativa (config.DATASET_FORMAT força a escolha)."""
    if config.DATASET_FORMAT and config.DATASET_FORMAT != "auto":
        return config.DATASET_FORMAT
    if list(config.DATASET_DIR.glob("feedbacks_*.json")):
        return "feedbacks"
    try:
        _encontrar_json_juiz()
        return "judge_json"
    except FileNotFoundError:
        pass
    raise FileNotFoundError(
        f"Não foi possível detectar o formato da fonte '{config.DATASET}' em "
        f"{config.DATASET_DIR}. Esperado feedbacks_*.json (formato 'feedbacks') "
        f"ou um JSON de questões com dificuldade do juiz (formato 'judge_json'). "
        f"Você também pode forçar via DATASET_FORMAT no .env."
    )


def build_dataset(
    usar_exemplos: bool | None = None,
    estrategia: str | None = None,
    salvar: bool = True,
) -> pd.DataFrame:
    """Monta a base completa (uma linha por questão) e salva o questoes.csv.

    Detecta o formato da fonte ativa e despacha para o montador adequado. A
    estratégia de rótulo (media/moda) só se aplica ao formato 'feedbacks'.
    """
    usar_exemplos = config.USE_EXAMPLES if usar_exemplos is None else usar_exemplos
    estrategia = config.LABEL_STRATEGY if estrategia is None else estrategia

    formato = _detectar_formato()
    print(f"Fonte ativa: {config.DATASET}  (formato: {formato})\n")

    if formato == "feedbacks":
        return _build_feedbacks(usar_exemplos, estrategia, salvar)
    if formato == "judge_json":
        return _build_judge_json(usar_exemplos, salvar)
    raise ValueError(
        f"Formato de fonte desconhecido: {formato!r}. "
        "Use 'auto', 'feedbacks' ou 'judge_json' em DATASET_FORMAT."
    )


def ensure_dataset() -> Path:
    """Garante que data/raw/<DATASET>/questoes.csv exista (gera se faltar)."""
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
        help=f"como agregar as notas no rótulo, só p/ fontes 'feedbacks' "
             f"(padrão: {config.LABEL_STRATEGY})",
    )
    args = parser.parse_args()
    build_dataset(
        usar_exemplos=False if args.sem_exemplos else None,
        estrategia=args.estrategia,
    )


if __name__ == "__main__":
    main()
