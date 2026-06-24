"""Etapa 4 — Baseline com LLM (Groq / Llama).

Classifica a dificuldade dos enunciados diretamente com o LLM, em modo
zero-shot ou few-shot, para comparar com os modelos de ML tradicionais.

Tem **cache incremental + retomada automática** (chave: ``config.ID_COL``):
cada lote é salvo em ``config.LLM_BASELINE_PREDS`` assim que termina, de modo
que estourar a cota da API no meio do caminho NÃO perde o trabalho já feito —
basta rodar de novo (no dia seguinte, ou com outro provedor) que o script pula
as questões já classificadas e continua de onde parou.

Pré-requisito: chave do provedor de LLM no .env e (para o few-shot) dataset_limpo.csv.

Uso:
    python -m src.llm_baseline --n 30              # 30 amostras do teste
    python -m src.llm_baseline --n 30 --few-shot
    python -m src.llm_baseline --n 0               # TODO o conjunto de teste (com retomada)
"""
from __future__ import annotations

import argparse
import time
import unicodedata
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score
from tqdm import tqdm

from . import config, data_utils, llm_client

SYSTEM = (
    "Você é um juiz experiente de maratonas de programação competitiva. "
    "Sua tarefa é estimar a dificuldade de questões a partir do enunciado."
)


def _normalize(label: str) -> str:
    """Mapeia variações ('muito fácil', 'very easy', 'hard'...) p/ o rótulo canônico."""
    txt = unicodedata.normalize("NFKD", str(label).lower())
    txt = "".join(c for c in txt if not unicodedata.combining(c)).strip()
    txt = txt.replace("_", " ").replace("-", " ")
    txt = " ".join(txt.split())   # colapsa espaços

    mapa = {
        "muito facil": "muito_facil", "very easy": "muito_facil",
        "muito baixa": "muito_facil", "muito simples": "muito_facil",
        "facil": "facil", "easy": "facil", "baixa": "facil", "simples": "facil",
        "medio": "medio", "media": "medio", "medium": "medio", "moderada": "medio",
        "muito dificil": "muito_dificil", "very hard": "muito_dificil",
        "muito alta": "muito_dificil", "muito complexa": "muito_dificil",
        "dificil": "dificil", "hard": "dificil", "alta": "dificil", "complexa": "dificil",
    }
    if txt in mapa:
        return mapa[txt]
    # Casamento parcial: testar os termos "muito ..." antes dos simples, para
    # não confundir "muito dificil" com "dificil".
    ordem = [
        "muito facil", "very easy", "muito dificil", "very hard",
        "facil", "easy", "medio", "medium", "media", "dificil", "hard",
    ]
    for chave in ordem:
        if chave in txt:
            return mapa[chave]
    return txt.replace(" ", "_")


def _build_prompt(statement: str, examples: list[tuple[str, str]] | None = None) -> str:
    partes: list[str] = []
    if examples:
        partes.append("Exemplos de referência (enunciado -> dificuldade):")
        for texto, rotulo in examples:
            partes.append(f'- "{texto[:400]}" -> {rotulo}')
        partes.append("")
    niveis = ", ".join(config.DIFFICULTY_LABELS)
    partes.append(
        f"Classifique a dificuldade do enunciado abaixo em exatamente um nível "
        f"entre: {niveis}.\n"
        'Responda APENAS em JSON, no formato: {"dificuldade": "<nivel>"}.\n\n'
        f"Enunciado:\n{statement[:4000]}"
    )
    return "\n".join(partes)


def classify_one(statement: str, examples: list[tuple[str, str]] | None = None) -> str:
    """Classifica um único enunciado via LLM e retorna o rótulo canônico."""
    data = llm_client.generate_json(_build_prompt(statement, examples), SYSTEM)
    if isinstance(data, dict):
        return _normalize(data.get("dificuldade", ""))
    return _normalize(str(data))


def _build_prompt_lote(
    statements: list[str],
    examples: list[tuple[str, str]] | None = None,
    max_chars: int = 1500,
) -> str:
    """Monta um prompt que classifica VÁRIOS enunciados de uma só vez."""
    partes: list[str] = []
    if examples:
        partes.append("Exemplos de referência (enunciado -> dificuldade):")
        for texto, rotulo in examples:
            partes.append(f'- "{texto[:300]}" -> {rotulo}')
        partes.append("")
    niveis = ", ".join(config.DIFFICULTY_LABELS)
    partes.append(
        f"Classifique a dificuldade de CADA enunciado abaixo em exatamente um "
        f"nível entre: {niveis}.\n"
        "Responda APENAS em JSON: um array com um objeto por enunciado, no "
        'formato [{"id": <numero>, "dificuldade": "<nivel>"}]. '
        "Inclua TODOS os ids e nada fora do JSON."
    )
    for i, texto in enumerate(statements, 1):
        partes.append(f"\nEnunciado {i}:\n{str(texto)[:max_chars]}")
    return "\n".join(partes)


def classify_batch(statements: list[str], examples: list[tuple[str, str]] | None = None) -> list[str]:
    """Classifica vários enunciados em UMA chamada à API (economiza cota).

    A resposta é casada por ``id``. Qualquer item que a resposta não cobrir
    (id ausente/ilegível) é reclassificado individualmente, garantindo que o
    resultado tenha o mesmo tamanho e a mesma corretude do modo item-a-item.
    """
    data = llm_client.generate_json(_build_prompt_lote(statements, examples), SYSTEM)
    por_id: dict[int, str] = {}
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict) and "id" in item:
                try:
                    idx = int(item["id"])
                except (TypeError, ValueError):
                    continue
                por_id[idx] = _normalize(item.get("dificuldade", ""))
    preds: list[str] = []
    for i in range(1, len(statements) + 1):
        rotulo = por_id.get(i)
        preds.append(rotulo if rotulo else classify_one(str(statements[i - 1]), examples))
    return preds


def few_shot_examples(df: pd.DataFrame, por_classe: int = 1) -> list[tuple[str, str]]:
    """Sorteia exemplos rotulados (um ou mais por classe) para o few-shot."""
    exemplos: list[tuple[str, str]] = []
    for classe in config.DIFFICULTY_LABELS:
        sub = df[df[config.LABEL_COL].astype(str).str.lower() == classe]
        if len(sub):
            amostra = sub.sample(min(por_classe, len(sub)), random_state=config.RANDOM_SEED)
            for _, row in amostra.iterrows():
                exemplos.append((str(row[config.TEXT_COL]), classe))
    return exemplos


def classify_series(textos, examples=None, sleep: float = 0.0, lote: int = 1) -> list[str]:
    """Classifica uma sequência de enunciados (com barra de progresso).

    ``lote`` > 1 agrupa os enunciados em lotes de ``lote`` por requisição
    (prompt packing), reduzindo o nº de chamadas à API ~``lote`` vezes.

    *Sem cache* — use ``classify_with_cache`` quando precisar de retomada.
    """
    textos = [str(t) for t in textos]
    if lote and lote > 1:
        preds: list[str] = []
        grupos = [textos[i:i + lote] for i in range(0, len(textos), lote)]
        for grupo in tqdm(grupos, desc=f"LLM (baseline, lote={lote})"):
            preds.extend(classify_batch(grupo, examples))
            if sleep:
                time.sleep(sleep)
        return preds
    preds = []
    for texto in tqdm(textos, desc="LLM (baseline)"):
        preds.append(classify_one(texto, examples))
        if sleep:
            time.sleep(sleep)
    return preds


# ----------------------------------------------------------------------------
# Cache incremental + retomada
# ----------------------------------------------------------------------------
def _carregar_cache(path: Path) -> pd.DataFrame:
    """Lê o CSV de cache se existir, garantindo a coluna pred_llm."""
    if path.exists():
        cache = pd.read_csv(path)
        if "pred_llm" in cache.columns:
            return cache
    return pd.DataFrame()


def _ids_em_cache(cache: pd.DataFrame) -> set[str]:
    if cache.empty or config.ID_COL not in cache.columns:
        return set()
    return set(cache[config.ID_COL].astype(str))


def _salvar_cache(cache: pd.DataFrame, path: Path) -> None:
    """Grava o cache de forma atômica (escreve em .tmp e renomeia)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    cache.to_csv(tmp, index=False)
    tmp.replace(path)


def classify_with_cache(
    df: pd.DataFrame,
    examples: list[tuple[str, str]] | None = None,
    *,
    sleep: float = 0.0,
    lote: int = 1,
    save_path: Path | None = None,
    text_col: str | None = None,
    id_col: str | None = None,
) -> pd.Series:
    """Classifica ``df`` com retomada: pula ids já presentes em ``save_path``.

    A cada lote processado, o cache CSV é regravado por inteiro (operação
    atômica via .tmp). Se a API estourar a cota no meio, o trabalho já feito
    está salvo — basta rodar o comando de novo (mesmo dia com outro provedor,
    ou no dia seguinte) que ele continua de onde parou.

    Retorna uma ``Series`` de predições alinhada à ordem de ``df``.
    """
    text_col = text_col or config.TEXT_COL
    id_col = id_col or config.ID_COL
    save_path = save_path or config.LLM_BASELINE_PREDS

    if id_col not in df.columns:
        raise KeyError(f"Coluna de id '{id_col}' não está no DataFrame de entrada.")

    cache = _carregar_cache(save_path)
    feitos = _ids_em_cache(cache)

    pendentes = df[~df[id_col].astype(str).isin(feitos)].copy()
    total = len(df)
    ja_cacheados = total - len(pendentes)
    if ja_cacheados:
        print(f"[cache] {ja_cacheados}/{total} já classificados — retomando do restante.")

    if not pendentes.empty:
        registros = pendentes.to_dict("records")
        lote_eff = max(1, int(lote or 1))
        grupos = [registros[i:i + lote_eff] for i in range(0, len(registros), lote_eff)]
        desc = f"LLM (baseline, lote={lote_eff})" if lote_eff > 1 else "LLM (baseline)"
        for grupo in tqdm(grupos, desc=desc):
            textos = [str(r[text_col]) for r in grupo]
            if lote_eff > 1:
                preds_lote = classify_batch(textos, examples)
            else:
                preds_lote = [classify_one(textos[0], examples)]
            novas_linhas = []
            for r, pred in zip(grupo, preds_lote):
                linha = dict(r)
                linha["pred_llm"] = pred
                novas_linhas.append(linha)
            cache = pd.concat([cache, pd.DataFrame(novas_linhas)], ignore_index=True)
            _salvar_cache(cache, save_path)
            if sleep:
                time.sleep(sleep)

    # Alinha as predições à ordem do df de entrada.
    mapa = dict(zip(cache[id_col].astype(str), cache["pred_llm"]))
    return df[id_col].astype(str).map(mapa)


def run(n: int = 30, few_shot: bool = False, sleep: float = 0.0, lote: int = 1) -> None:
    """Roda o baseline LLM sobre uma amostra (ou TODO o teste, se n<=0).

    Reaproveita ``config.LLM_BASELINE_PREDS`` como cache: se a cota da API
    estourar, basta rodar de novo que continua de onde parou.
    """
    config.require_api_key()
    if not config.CLEAN_DATASET.exists():
        raise FileNotFoundError(
            "dataset_limpo.csv não encontrado. Rode antes: python -m src.preprocess"
        )

    df = pd.read_csv(config.CLEAN_DATASET)
    df_train, df_test = data_utils.split_train_test(df)

    if n is None or n <= 0:
        amostra = df_test
        print(f"Usando TODO o conjunto de teste ({len(amostra)} exemplos).")
    else:
        amostra = df_test.sample(min(n, len(df_test)), random_state=config.RANDOM_SEED)

    exemplos = few_shot_examples(df_train, por_classe=1) if few_shot else None
    if exemplos:
        print(f"Few-shot com {len(exemplos)} exemplo(s).")
    if lote > 1:
        print(f"Lote (prompt packing): {lote} enunciados por requisição.")

    preds = classify_with_cache(
        amostra,
        examples=exemplos,
        sleep=sleep,
        lote=lote,
        save_path=config.LLM_BASELINE_PREDS,
    )
    amostra = amostra.copy()
    amostra["pred_llm"] = preds.values
    print(f"\nPredições no cache: {config.LLM_BASELINE_PREDS}")

    if config.LABEL_COL in amostra.columns:
        y_true = amostra[config.LABEL_COL].astype(str).str.lower()
        acc = accuracy_score(y_true, preds)
        f1 = f1_score(y_true, preds, average="macro", zero_division=0)
        print(f"\nAcurácia: {acc:.3f} | F1 macro: {f1:.3f}\n")
        print(classification_report(y_true, preds, zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 4 - Baseline de classificação com LLM (Groq)")
    parser.add_argument("--n", type=int, default=30,
                        help="qtd. de exemplos do teste (0 = todo o conjunto de teste, com retomada)")
    parser.add_argument("--few-shot", action="store_true", help="inclui exemplos no prompt")
    parser.add_argument("--sleep", type=float, default=0.0, help="pausa entre chamadas (s)")
    parser.add_argument("--lote", type=int, default=10,
                        help="enunciados por requisição (prompt packing; >1 economiza cota). "
                             "Padrão 10 — reduz ~10x as chamadas à API. Use --lote 1 p/ item-a-item.")
    args = parser.parse_args()
    run(n=args.n, few_shot=args.few_shot, sleep=args.sleep, lote=args.lote)


if __name__ == "__main__":
    main()
