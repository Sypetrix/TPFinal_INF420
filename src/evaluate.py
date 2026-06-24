"""Etapa 7 — Avaliação final comparativa.

Compara, sobre a MESMA amostra de teste, três abordagens:
  (A) ML puro            -> features TF-IDF
  (B) LLM puro           -> baseline direto com Groq/Llama
  (C) ML + features LLM  -> TF-IDF concatenado aos conceitos extraídos (Etapa 5)

Para (A) e (C) usa-se o mesmo estimador (Regressão Logística), de modo que a
única diferença seja o conjunto de features — tornando a comparação justa.

O TF-IDF é ajustado **apenas com o conjunto de treino** (não o vetorizador salvo
pela Etapa 2, que vê toda a base), para que a comparação não sofra vazamento de
informação do teste. A amostra de teste é fixa porque a abordagem (B) chama a
API do LLM e tem custo (cota) — daí não usar validação cruzada aqui.

Pré-requisitos:
  - python -m src.preprocess        (sempre)
  - python -m src.llm_concepts      (necessário para a abordagem C)
  - GROQ_API_KEY no .env            (necessário para a abordagem B)

Uso:
    python -m src.evaluate --n 40            # compara A, B e C
    python -m src.evaluate --n 40 --no-llm   # compara só A e C (sem chamar a API)
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from . import config, data_utils
from .llm_concepts import CONCEITOS, ROW_KEY


def _metricas(y_true, y_pred) -> dict:
    return {
        "acuracia": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_ponderado": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def _matriz_conceitos(df_sub: pd.DataFrame, feats: pd.DataFrame) -> csr_matrix:
    """Alinha as features de conceitos (por _row) na ordem de df_sub."""
    alinhado = (
        feats.set_index(ROW_KEY)
        .reindex(df_sub[ROW_KEY])[CONCEITOS]
        .fillna(0)
        .to_numpy()
    )
    return csr_matrix(alinhado)


def run(n_sample: int = 40, usar_llm: bool = True, lote: int = 1, sleep: float = 0.0) -> None:
    if not config.CLEAN_DATASET.exists():
        raise FileNotFoundError(
            "dataset_limpo.csv não encontrado. Rode antes: python -m src.preprocess"
        )

    df = pd.read_csv(config.CLEAN_DATASET)
    df = df.dropna(subset=["texto_limpo", config.LABEL_COL]).reset_index(drop=True)
    df[ROW_KEY] = df.index

    df_train, df_test = data_utils.split_train_test(df)
    if n_sample is None or n_sample <= 0:
        amostra = df_test
        print(f"Avaliação sobre TODO o conjunto de teste ({len(amostra)} exemplos).")
    else:
        amostra = df_test.sample(min(n_sample, len(df_test)), random_state=config.RANDOM_SEED)

    y_train = df_train[config.LABEL_COL].astype(str).str.lower()
    y_eval = amostra[config.LABEL_COL].astype(str).str.lower()

    # TF-IDF ajustado SÓ no treino (evita vazamento de informação do teste).
    vectorizer = data_utils.build_vectorizer()
    Xtr = vectorizer.fit_transform(df_train["texto_limpo"].astype(str))
    Xev = vectorizer.transform(amostra["texto_limpo"].astype(str))

    resultados: list[dict] = []

    # ---- (A) ML puro (TF-IDF) ----
    clf_a = LogisticRegression(max_iter=1000, class_weight="balanced",
                               random_state=config.RANDOM_SEED)
    clf_a.fit(Xtr, y_train)
    pred_a = clf_a.predict(Xev)
    resultados.append({"abordagem": "A) ML puro (TF-IDF)", **_metricas(y_eval, pred_a)})

    # ---- (C) ML + features extraídas pela LLM ----
    if config.LLM_FEATURES.exists():
        feats = pd.read_csv(config.LLM_FEATURES)
        Ftr = _matriz_conceitos(df_train, feats)
        Fev = _matriz_conceitos(amostra, feats)
        Xtr_c = hstack([Xtr, Ftr])
        Xev_c = hstack([Xev, Fev])
        clf_c = LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=config.RANDOM_SEED)
        clf_c.fit(Xtr_c, y_train)
        pred_c = clf_c.predict(Xev_c)
        resultados.append({"abordagem": "C) ML + features LLM", **_metricas(y_eval, pred_c)})
    else:
        print("[C] pulada: rode 'python -m src.llm_concepts' para gerar llm_features.csv.\n")

    # ---- (B) LLM puro (Groq/Llama) ----
    if usar_llm:
        from . import llm_baseline   # import preguiçoso (depende do pacote do provedor)

        exemplos = llm_baseline.few_shot_examples(df_train, por_classe=1)
        # Usa o cache incremental: se a cota estourar no meio, basta rodar de
        # novo que continua de onde parou (pula ids já em llm_baseline_preds.csv).
        pred_b = llm_baseline.classify_with_cache(
            amostra,
            examples=exemplos,
            sleep=sleep,
            lote=lote,
            save_path=config.LLM_BASELINE_PREDS,
        )
        # Se a cota acabar antes do fim, alguns ids ficam sem predição (NaN).
        # Avalia só os que têm predição (e avisa quantos faltam).
        mask = pred_b.notna()
        faltando = int((~mask).sum())
        if faltando:
            print(f"[B] {faltando}/{len(amostra)} ainda sem predição "
                  f"(cota da API esgotada?). Métricas calculadas sobre "
                  f"{int(mask.sum())} exemplos já no cache; rode de novo "
                  f"para terminar.")
        if mask.any():
            resultados.append({"abordagem": "B) LLM puro (Groq)",
                               **_metricas(y_eval[mask.values], pred_b[mask].astype(str))})
    else:
        print("[B] pulada: execução com --no-llm.\n")

    tabela = pd.DataFrame(resultados).sort_values("f1_macro", ascending=False)
    tabela.to_csv(config.MODELS_DIR / "comparacao_final.csv", index=False)

    print("=" * 60)
    print(f"COMPARAÇÃO FINAL (amostra de teste: {len(amostra)} exemplos)")
    print("=" * 60)
    print(tabela.to_string(index=False))
    print(f"\nTabela salva em: {config.MODELS_DIR / 'comparacao_final.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 7 - Avaliação final comparativa")
    parser.add_argument("--n", type=int, default=40,
                        help="tamanho da amostra de teste (0 = todo o conjunto de teste, com retomada)")
    parser.add_argument("--no-llm", action="store_true", help="não chama a API do LLM/Groq (pula B)")
    parser.add_argument("--lote", type=int, default=10,
                        help="enunciados por requisição no baseline LLM (prompt packing). "
                             "Padrão 10 — reduz ~10x as chamadas à API. Use --lote 1 p/ item-a-item.")
    parser.add_argument("--sleep", type=float, default=0.0,
                        help="pausa entre chamadas à API (s) — ajuda a não estourar o limite por minuto")
    args = parser.parse_args()
    run(n_sample=args.n, usar_llm=not args.no_llm, lote=args.lote, sleep=args.sleep)


if __name__ == "__main__":
    main()
