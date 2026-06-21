"""Inferência — avaliar questões NOVAS (sem rótulo de dificuldade).

Esta é a etapa que coloca o projeto "em produção": dado um conjunto de questões
ainda **não rotuladas** (ex.: as questões que um professor quer avaliar), a
ferramenta, para cada questão:

  1. **prevê a dificuldade** com o classificador de ML já treinado
     (``models/<DATASET>/best_ml_model.joblib``) — offline, sem API;
  2. **identifica o(s) tópico(s)/conceito(s)** (if/else, repetição, grafos,
     caminho mínimo, etc.) usando o LLM (Groq/Llama) — opcional (``--no-llm``);
  3. **recomenda questões similares** do banco já existente (o catálogo
     multi-fonte do recomendador), úteis como referência/treino.

Dois modos de entrada:

  * ``--fonte <Nome>``  : lê ``data/raw/<Nome>/questoes.csv`` (consolidado pela
    Etapa 1). É o uso "professor": basta criar ``arquivos/<Nome>/`` com um JSON
    no mesmo formato das demais fontes (ex.: uma pasta ``avaliar/``), rodar
    ``DATASET=<Nome> python -m src.ingest`` e depois apontar aqui.
  * ``--teste``         : amostra algumas questões SEM rótulo já presentes no
    catálogo do recomendador (SPOJ/OBI/Neps/INF110) para testar a ferramenta
    ponta a ponta, sem precisar de questões externas.

Pré-requisitos:
  - ML (sempre):   modelo treinado em ``models/<DATASET>/`` (rode antes
    ``python -m src.preprocess`` e ``python -m src.train_ml``);
  - LLM (opcional): GROQ_API_KEY no .env (omita com ``--no-llm``).

Uso:
    python -m src.predict --teste --n 5 --no-llm        # demonstração offline
    python -m src.predict --teste --n 5                 # demonstração completa (usa API)
    python -m src.predict --fonte avaliar --top-k 3     # avalia uma fonte nova
    python -m src.predict --fonte SPOJ --n 10 --lote 5  # avalia 10 questões do SPOJ
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd

from . import config, data_utils
from .recommend import RecomendadorConteudo


# ----------------------------------------------------------------------------
# Seleção das questões a avaliar
# ----------------------------------------------------------------------------
def _questoes_da_fonte(fonte: str, so_nao_rotuladas: bool, n: int | None) -> pd.DataFrame:
    """Lê data/raw/<fonte>/questoes.csv e devolve as questões a avaliar."""
    csv = config.questoes_csv_for(fonte)
    if not csv.exists():
        raise SystemExit(
            f"Base da fonte '{fonte}' não encontrada em {csv}.\n"
            f"Rode antes: DATASET={fonte} python -m src.ingest"
        )
    df = pd.read_csv(csv)
    if config.LABEL_COL not in df.columns:
        df[config.LABEL_COL] = pd.NA
    df["fonte"] = fonte
    if so_nao_rotuladas:
        df = df[df[config.LABEL_COL].isna()]
        if df.empty:
            raise SystemExit(f"A fonte '{fonte}' não tem questões sem rótulo.")
    df = df.dropna(subset=[config.TEXT_COL])
    if n:
        df = df.head(n)
    return df.reset_index(drop=True)


def _questoes_de_teste(rec: RecomendadorConteudo, n: int) -> pd.DataFrame:
    """Amostra N questões SEM rótulo do catálogo do recomendador (modo --teste)."""
    nao_rot = rec.df[rec.df[config.LABEL_COL].isna()]
    if nao_rot.empty:
        raise SystemExit(
            "Não há questões sem rótulo no catálogo do recomendador "
            f"(fontes: {', '.join(rec.fontes)})."
        )
    return nao_rot.sample(min(n, len(nao_rot)), random_state=config.RANDOM_SEED)


# ----------------------------------------------------------------------------
# Previsões
# ----------------------------------------------------------------------------
def _prever_dificuldade_ml(textos: list[str]) -> list[str]:
    """Prevê a dificuldade com o Pipeline de ML treinado (offline, sem API)."""
    if not config.BEST_ML_MODEL.exists():
        raise SystemExit(
            f"Modelo de ML não encontrado em {config.BEST_ML_MODEL}.\n"
            "Rode antes: python -m src.preprocess && python -m src.train_ml"
        )
    modelo = joblib.load(config.BEST_ML_MODEL)
    # O Pipeline (TF-IDF + classificador) foi treinado sobre 'texto_limpo', então
    # aplicamos a MESMA limpeza aos enunciados novos antes de prever.
    limpos = [data_utils.clean_text(str(t)) for t in textos]
    return [str(p) for p in modelo.predict(limpos)]


def _topicos(vetor: dict[str, int]) -> list[str]:
    """Converte {conceito: 0/1} na lista de conceitos presentes."""
    return [c for c, v in vetor.items() if v]


def _extrair_topicos(textos: list[str], lote: int) -> list[list[str]]:
    """Identifica os tópicos/conceitos de cada enunciado via LLM (Groq)."""
    from . import llm_features  # import preguiçoso (depende do pacote `groq`)

    if lote and lote > 1:
        vetores: list[dict[str, int]] = []
        for i in range(0, len(textos), lote):
            vetores.extend(llm_features.extract_batch(textos[i:i + lote]))
    else:
        vetores = [llm_features.extract_one(t) for t in textos]
    return [_topicos(v) for v in vetores]


def _prever_dificuldade_llm(textos: list[str], lote: int) -> list[str]:
    """Prevê a dificuldade via LLM (Groq), com few-shot se houver base rotulada."""
    from . import llm_baseline  # import preguiçoso (depende do pacote `groq`)

    exemplos = None
    if config.CLEAN_DATASET.exists():
        base = pd.read_csv(config.CLEAN_DATASET)
        if config.LABEL_COL in base.columns:
            rotuladas = base.dropna(subset=[config.LABEL_COL])
            if len(rotuladas):
                exemplos = llm_baseline.few_shot_examples(rotuladas, por_classe=1)
    return llm_baseline.classify_series(textos, exemplos, lote=lote)


# ----------------------------------------------------------------------------
# Orquestração
# ----------------------------------------------------------------------------
def _resumo(texto: str, n: int = 90) -> str:
    txt = " ".join(str(texto).split())
    return txt[:n] + ("…" if len(txt) > n else "")


def run(
    fonte: str | None = None,
    teste: bool = False,
    n: int | None = None,
    so_nao_rotuladas: bool = False,
    usar_llm: bool = True,
    top_k: int = 3,
    lote: int = 1,
    ignorar_nivel: bool = False,
) -> None:
    # Falha rápido se a etapa LLM foi pedida sem chave (antes de processar nada).
    if usar_llm:
        config.require_api_key()

    # O recomendador também serve de catálogo de referência (banco existente).
    rec = RecomendadorConteudo()
    print(f"Catálogo de referência: {len(rec.df)} questões de {', '.join(rec.fontes)}\n")

    if teste:
        questoes = _questoes_de_teste(rec, n or 5)
        print(f"[modo teste] {len(questoes)} questão(ões) SEM rótulo amostrada(s) do catálogo.\n")
    else:
        if not fonte:
            raise SystemExit("Informe --fonte <Nome> ou use --teste.")
        questoes = _questoes_da_fonte(fonte, so_nao_rotuladas, n)
        print(f"[fonte {fonte}] {len(questoes)} questão(ões) a avaliar.\n")

    textos = [str(t) for t in questoes[config.TEXT_COL].tolist()]
    ids = questoes[config.ID_COL].tolist() if config.ID_COL in questoes.columns else list(range(len(textos)))
    fontes = questoes["fonte"].tolist() if "fonte" in questoes.columns else [fonte or "?"] * len(textos)

    # 1) Dificuldade via ML (sempre)
    print("Prevendo dificuldade com o modelo de ML treinado...")
    dif_ml = _prever_dificuldade_ml(textos)

    # 2) Dificuldade + tópicos via LLM (opcional)
    dif_llm: list[str | None] = [None] * len(textos)
    topicos: list[list[str]] = [[] for _ in textos]
    if usar_llm:
        print("Identificando tópicos/conceitos via LLM (Groq)...")
        topicos = _extrair_topicos(textos, lote)
        print("Prevendo dificuldade via LLM (Groq)...")
        dif_llm = _prever_dificuldade_llm(textos, lote)
    else:
        print("[--no-llm] pulando tópicos e dificuldade via LLM (só ML).")

    # 3) Recomendações do banco: questões SEMELHANTES (similaridade de conteúdo) e,
    #    por padrão, no MESMO nível de dificuldade previsto — este é o elo
    #    classificação -> recomendação (a dificuldade/conceitos identificados guiam
    #    quais exercícios do banco rotulado são sugeridos). Use --ignorar-nivel para
    #    recomendar só por conteúdo, sem o filtro de nível.
    print("Buscando questões similares no banco...\n")
    linhas: list[dict] = []
    for i, (qid, fnt, texto) in enumerate(zip(ids, fontes, textos)):
        excluir = rec.df.index[
            (rec.df[config.ID_COL].astype(str) == str(qid)) & (rec.df["fonte"] == fnt)
        ].tolist()
        # Nível-alvo = dificuldade prevista (LLM se houver, senão ML).
        nivel = None if ignorar_nivel else (dif_llm[i] or dif_ml[i])
        recs = rec.recomendar_por_texto(texto, nivel_alvo=nivel, top_k=top_k, excluir_idx=excluir)
        if nivel and recs.empty:   # nível sem candidatos -> cai para conteúdo puro
            recs = rec.recomendar_por_texto(texto, top_k=top_k, excluir_idx=excluir)
            nivel = None
        criterio = f"nível previsto: {nivel}" if nivel else "conteúdo (sem filtro de nível)"
        rec_str = "; ".join(
            f"{r[config.ID_COL]}({r['fonte']})" for _, r in recs.iterrows()
        )

        print("=" * 78)
        print(f"Questão {qid} [{fnt}]")
        print(f"  Enunciado     : {_resumo(texto)}")
        print(f"  Dificuldade ML: {dif_ml[i]}")
        if usar_llm:
            print(f"  Dificuldade LLM: {dif_llm[i]}")
            print(f"  Tópicos (LLM) : {', '.join(topicos[i]) if topicos[i] else '(nenhum identificado)'}")
        print(f"  Recomendações ({criterio}): {rec_str if rec_str else '(nenhuma)'}")
        for _, r in recs.iterrows():
            nivel = r.get(config.LABEL_COL)
            nivel = "" if pd.isna(nivel) else f" [{nivel}]"
            print(f"      - {r[config.ID_COL]} ({r['fonte']}){nivel}  sim={r['similaridade']:.3f}  {_resumo(r[config.TEXT_COL], 60)}")

        linhas.append({
            config.ID_COL: qid,
            "fonte": fnt,
            "dificuldade_ml": dif_ml[i],
            "dificuldade_llm": dif_llm[i],
            "topicos": ", ".join(topicos[i]),
            "rec_criterio": criterio,
            "recomendacoes": rec_str,
        })

    saida = config.PROCESSED_DIR / "predicoes.csv"
    pd.DataFrame(linhas).to_csv(saida, index=False)
    print("=" * 78)
    print(f"\nPredições salvas em: {saida}")
    if teste:
        print(
            "\nObs.: no modo --teste as questões não têm rótulo verdadeiro, então "
            "isto demonstra o FUNCIONAMENTO ponta a ponta (não a acurácia). A "
            "acurácia do classificador é medida na Etapa 7 (src.evaluate), sobre "
            "dados rotulados."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inferência - prevê dificuldade + tópico e recomenda (questões novas)"
    )
    parser.add_argument("--fonte", type=str, default=None,
                        help="fonte consolidada a avaliar (data/raw/<fonte>/questoes.csv)")
    parser.add_argument("--teste", action="store_true",
                        help="amostra questões sem rótulo do catálogo do recomendador")
    parser.add_argument("--n", type=int, default=None, help="qtd. de questões a avaliar")
    parser.add_argument("--so-nao-rotuladas", action="store_true",
                        help="no modo --fonte, avalia só as questões sem rótulo")
    parser.add_argument("--no-llm", action="store_true",
                        help="não chama a API do LLM/Groq (só dificuldade via ML)")
    parser.add_argument("--top-k", type=int, default=3, help="nº de recomendações por questão")
    parser.add_argument("--lote", type=int, default=1,
                        help="enunciados por requisição no LLM (prompt packing; >1 economiza cota)")
    parser.add_argument("--ignorar-nivel", action="store_true",
                        help="recomenda só por conteúdo, sem filtrar pelo nível previsto")
    args = parser.parse_args()
    run(
        fonte=args.fonte,
        teste=args.teste,
        n=args.n,
        so_nao_rotuladas=args.so_nao_rotuladas,
        usar_llm=not args.no_llm,
        top_k=args.top_k,
        lote=args.lote,
        ignorar_nivel=args.ignorar_nivel,
    )


if __name__ == "__main__":
    main()
