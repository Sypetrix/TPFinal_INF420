"""Aplica o modelo treinado às questões a avaliar (arquivos/avaliar) — uso-fim.

Dado um conjunto de questões **sem rótulo** (ex.: as de ``arquivos/avaliar/``), a
ferramenta, para cada questão:

  1. **prevê a dificuldade** com o classificador de ML treinado na base rotulada
     (``models/<DATASET>/best_ml_model[_3niveis].joblib``) — offline, sem API. A
     dificuldade é decidida pelo **ML**, não pela LLM (papel 2 do contexto.md);
  2. **identifica os conceitos** (if/else, repetição, grafos, programação
     dinâmica…) com a LLM — função central, opcional (``--no-llm``);
  3. **recomenda questões** do banco por **conceitos em comum + dificuldade
     compatível** (mesmo nível ±1), via ``src.recommend``;
  4. *(opcional, ``--explicar``)* gera, com a LLM, a justificativa da recomendação.

Modos de entrada:
  * ``--fonte <Nome>``  : lê ``data/raw/<Nome>/questoes.csv`` (uso "professor":
    crie ``arquivos/<Nome>/`` com um JSON judge_json, ex. ``avaliar``);
  * ``--teste``         : amostra questões sem rótulo já no catálogo (demo).

Pré-requisitos:
  - ML (sempre):   ``python -m src.preprocess`` + ``python -m src.train_ml``
    (use ``--niveis 3`` para o modelo de 3 níveis, padrão do produto);
  - LLM (opcional): chave do provedor no .env (LLM_PROVIDER); omita com ``--no-llm``.

Uso:
    python -m src.predict_difficulty --fonte avaliar --no-llm        # offline
    python -m src.predict_difficulty --fonte avaliar --lote 5        # completo
    python -m src.predict_difficulty --teste --n 5 --no-llm          # demo
    python -m src.predict_difficulty --fonte avaliar --explicar      # + justificativa
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd

from . import config, data_utils
from .recommend import RecomendadorConteudo, _norm_nivel


# ----------------------------------------------------------------------------
# Seleção das questões a avaliar
# ----------------------------------------------------------------------------
def _questoes_da_fonte(fonte: str, so_nao_rotuladas: bool, n: int | None) -> pd.DataFrame:
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
def _modelo_path(niveis: int):
    nome = "best_ml_model_3niveis.joblib" if niveis == 3 else "best_ml_model.joblib"
    return config.MODELS_DIR / nome


def _prever_dificuldade_ml(textos: list[str], niveis: int) -> list[str]:
    """Prevê a dificuldade com o Pipeline de ML treinado (offline, sem API)."""
    modelo_path = _modelo_path(niveis)
    if not modelo_path.exists():
        sufixo = " --niveis 3" if niveis == 3 else ""
        raise SystemExit(
            f"Modelo de ML não encontrado em {modelo_path}.\n"
            f"Rode antes: python -m src.preprocess && python -m src.train_ml{sufixo}"
        )
    modelo = joblib.load(modelo_path)
    # O Pipeline (TF-IDF + classificador) foi treinado sobre 'texto_limpo'; logo,
    # aplicamos a MESMA limpeza aos enunciados novos antes de prever.
    limpos = [data_utils.clean_text(str(t)) for t in textos]
    return [str(p) for p in modelo.predict(limpos)]


def _extrair_conceitos(textos: list[str], lote: int) -> list[list[str]]:
    """Identifica os conceitos de cada enunciado via LLM."""
    from . import llm_concepts  # import preguiçoso (depende do pacote `openai`)

    if lote and lote > 1:
        vetores: list[dict[str, int]] = []
        for i in range(0, len(textos), lote):
            vetores.extend(llm_concepts.extract_batch(textos[i:i + lote]))
    else:
        vetores = [llm_concepts.extract_one(t) for t in textos]
    return [[c for c, v in vet.items() if v] for vet in vetores]


def _relacao_nivel(rec: RecomendadorConteudo, nivel_q: str, nivel_r: str) -> str:
    """Descreve a relação de dificuldade entre a questão e a recomendada."""
    ia, ib = rec.idx_nivel.get(_norm_nivel(nivel_q)), rec.idx_nivel.get(_norm_nivel(nivel_r))
    if ia is None or ib is None:
        return "nível desconhecido"
    if ib < ia:
        return "um nível abaixo"
    if ib > ia:
        return "um nível acima"
    return "no mesmo nível"


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
    niveis: int = 3,
    ignorar_nivel: bool = False,
    explicar: bool = False,
    manter_duplicatas: bool = False,
) -> None:
    if usar_llm:
        config.require_api_key()   # falha rápido se faltar a chave
    dup = None if manter_duplicatas else 0.95   # descarta a mesma questão (quase idêntica)

    rec = RecomendadorConteudo(niveis=niveis)
    print(f"Catálogo: {len(rec.df)} questões | níveis: {niveis} | "
          f"conceitos no catálogo: {'sim' if rec.tem_conceitos else 'não (fallback TF-IDF)'}\n")

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

    # 1) Dificuldade — SÓ pelo modelo de ML (a LLM não classifica o avaliar)
    print("Prevendo dificuldade com o modelo de ML treinado...")
    dif = _prever_dificuldade_ml(textos, niveis)

    # 2) Conceitos — pela LLM (função central); opcional
    conceitos: list[list[str]] = [[] for _ in textos]
    if usar_llm:
        print("Identificando conceitos via LLM...")
        conceitos = _extrair_conceitos(textos, lote)
    else:
        print("[--no-llm] sem conceitos via LLM (recomendação cai para TF-IDF).")

    explicador = None
    if explicar:
        config.require_api_key()
        from . import llm_explain
        explicador = llm_explain

    # 3) Recomendação — conceitos em comum + dificuldade compatível (±1)
    print("Recomendando questões do banco (conceitos + dificuldade)...\n")
    linhas: list[dict] = []
    for i, (qid, fnt, texto) in enumerate(zip(ids, fontes, textos)):
        excluir = rec.df.index[
            (rec.df[config.ID_COL].astype(str) == str(qid)) & (rec.df["fonte"] == fnt)
        ].tolist()
        nivel = None if ignorar_nivel else dif[i]
        conj = set(conceitos[i]) or None
        recs = rec.recomendar_por_texto(texto, conceitos=conj, nivel_alvo=nivel,
                                        top_k=top_k, excluir_idx=excluir, dup_threshold=dup)
        if nivel and recs.empty:   # nível sem candidatos -> relaxa o filtro
            recs = rec.recomendar_por_texto(texto, conceitos=conj, top_k=top_k,
                                            excluir_idx=excluir, dup_threshold=dup)
        rec_str = "; ".join(f"{r[config.ID_COL]}({r['fonte']})" for _, r in recs.iterrows())

        print("=" * 78)
        print(f"Questão {qid} [{fnt}]")
        print(f"  Enunciado  : {_resumo(texto)}")
        print(f"  Dificuldade (ML): {dif[i]}")
        if usar_llm:
            print(f"  Conceitos (LLM) : {', '.join(conceitos[i]) if conceitos[i] else '(nenhum)'}")
        print(f"  Recomendações: {rec_str if rec_str else '(nenhuma)'}")
        for _, r in recs.iterrows():
            nv = r.get("dificuldade_efetiva")
            nv = "" if pd.isna(nv) else f" [{nv}]"
            print(f"      - {r[config.ID_COL]} ({r['fonte']}){nv}  score={r['score']:.3f}  "
                  f"{_resumo(r[config.TEXT_COL], 55)}")
            if r.get("conceitos"):
                print(f"          conceitos: {r['conceitos']}")
            if explicador is not None:
                comuns = sorted(set(conceitos[i]) & set(str(r.get("conceitos", "")).split(", ")) - {""})
                relacao = _relacao_nivel(rec, dif[i], r.get("dificuldade_efetiva"))
                txt = explicador.explicar_recomendacao(
                    texto, str(r[config.TEXT_COL]), comuns, relacao
                )
                print(f"          ↳ por quê: {txt}")

        linhas.append({
            config.ID_COL: qid,
            "fonte": fnt,
            "dificuldade": dif[i],
            "conceitos": ", ".join(conceitos[i]),
            "recomendacoes": rec_str,
        })

    saida = config.PROCESSED_DIR / "avaliar_classificado.csv"
    pd.DataFrame(linhas).to_csv(saida, index=False)
    print("=" * 78)
    print(f"\nResultado salvo em: {saida}")
    if teste:
        print(
            "\nObs.: no modo --teste as questões não têm rótulo verdadeiro — isto "
            "demonstra o FUNCIONAMENTO ponta a ponta, não a acurácia (medida na "
            "Etapa 7, src.evaluate, sobre dados rotulados)."
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aplica o modelo treinado a questões novas (dificuldade + conceitos + recomendação)"
    )
    parser.add_argument("--fonte", type=str, default=None,
                        help="fonte consolidada a avaliar (data/raw/<fonte>/questoes.csv)")
    parser.add_argument("--teste", action="store_true",
                        help="amostra questões sem rótulo do catálogo do recomendador")
    parser.add_argument("--n", type=int, default=None, help="qtd. de questões a avaliar")
    parser.add_argument("--so-nao-rotuladas", action="store_true",
                        help="no modo --fonte, avalia só as questões sem rótulo")
    parser.add_argument("--no-llm", action="store_true",
                        help="não chama a API do LLM (sem conceitos; recomendação por TF-IDF)")
    parser.add_argument("--top-k", type=int, default=3, help="nº de recomendações por questão")
    parser.add_argument("--lote", type=int, default=1,
                        help="enunciados por requisição no LLM (prompt packing; >1 economiza cota)")
    parser.add_argument("--niveis", type=int, choices=[5, 3], default=3,
                        help="granularidade da dificuldade (padrão 3; precisa do modelo treinado)")
    parser.add_argument("--ignorar-nivel", action="store_true",
                        help="recomenda sem filtrar pela dificuldade prevista")
    parser.add_argument("--explicar", action="store_true",
                        help="gera (via LLM) a justificativa de cada recomendação")
    parser.add_argument("--manter-duplicatas", action="store_true",
                        help="não descarta questões quase idênticas (úteis p/ ver gêmeas entre fontes)")
    args = parser.parse_args()
    run(
        fonte=args.fonte, teste=args.teste, n=args.n,
        so_nao_rotuladas=args.so_nao_rotuladas, usar_llm=not args.no_llm,
        top_k=args.top_k, lote=args.lote, niveis=args.niveis,
        ignorar_nivel=args.ignorar_nivel, explicar=args.explicar,
        manter_duplicatas=args.manter_duplicatas,
    )


if __name__ == "__main__":
    main()
