"""Etapa 3 — Modelos tradicionais de Aprendizado de Máquina.

Treina e compara quatro famílias de modelos sobre features TF-IDF:
Regressão Logística, KNN, SVM (linear) e Random Forest.

Metodologia de avaliação (pontos importantes):
  - **Validação cruzada estratificada** em vez de um único holdout. A base
    rotulada é pequena (dezenas de exemplos), então um único split de teste daria
    métricas muito instáveis; a CV usa todos os exemplos como teste uma vez e
    reporta média ± desvio entre folds.
  - **Sem vazamento de dados**: o TF-IDF entra num ``Pipeline`` e é reajustado
    DENTRO de cada fold (só com os dados de treino daquele fold), nunca vendo o
    conjunto de teste.
  - **Busca de hiperparâmetros** (``GridSearchCV``) pequena por família, para a
    comparação ser entre modelos bem ajustados, num mesmo cenário.
  - **Métricas**: acurácia, F1-macro e F1-ponderado (média entre folds) + uma
    matriz de confusão e um relatório por classe construídos com predições
    *out-of-fold* (honestas) do melhor modelo.

O melhor modelo (por F1-macro) é reajustado em TODA a base rotulada e salvo como
um ``Pipeline`` autossuficiente (TF-IDF + classificador), pronto para classificar
novos enunciados sem depender de um vetorizador externo.

Pré-requisito: rodar antes `python -m src.preprocess`.

Uso:
    python -m src.train_ml
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    make_scorer,
)
from sklearn.model_selection import GridSearchCV, cross_val_predict, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

from . import config, data_utils


def get_models() -> dict:
    """``{nome: (estimador, grade de hiperparâmetros)}`` das quatro famílias.

    As grades são pequenas de propósito: com poucos exemplos rotulados, buscas
    grandes só aumentariam a variância. Os nomes dos parâmetros levam o prefixo
    ``clf__`` porque o estimador entra dentro de um ``Pipeline`` com o TF-IDF.
    """
    seed = config.RANDOM_SEED
    return {
        "Regressao Logistica": (
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=seed),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "KNN": (
            KNeighborsClassifier(metric="cosine"),
            {"clf__n_neighbors": [3, 5, 7, 9], "clf__weights": ["uniform", "distance"]},
        ),
        "SVM (linear)": (
            LinearSVC(class_weight="balanced", random_state=seed, max_iter=5000),
            {"clf__C": [0.1, 1.0, 10.0]},
        ),
        "Random Forest": (
            RandomForestClassifier(class_weight="balanced", random_state=seed, n_jobs=-1),
            {"clf__n_estimators": [200, 400], "clf__max_depth": [None, 30]},
        ),
    }


# Métricas reportadas (média entre folds). F1 com zero_division=0 para não
# quebrar quando uma classe rara não é prevista em algum fold.
SCORING = {
    "acuracia": "accuracy",
    "f1_macro": make_scorer(f1_score, average="macro", zero_division=0),
    "f1_ponderado": make_scorer(f1_score, average="weighted", zero_division=0),
}


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def _baseline_majoritario(X, y, cv) -> dict:
    """Métricas de um classificador trivial (sempre a classe majoritária).

    Serve de piso de comparação: qualquer modelo útil precisa superá-lo,
    sobretudo em F1-macro (que penaliza ignorar as classes minoritárias).
    """
    pipe = Pipeline([
        ("tfidf", data_utils.build_vectorizer()),
        ("clf", DummyClassifier(strategy="most_frequent")),
    ])
    res = cross_validate(pipe, X, y, cv=cv, scoring=SCORING)
    return {
        "modelo": "Baseline (classe majoritária)",
        "acuracia": res["test_acuracia"].mean(),
        "acuracia_std": res["test_acuracia"].std(),
        "f1_macro": res["test_f1_macro"].mean(),
        "f1_macro_std": res["test_f1_macro"].std(),
        "f1_ponderado": res["test_f1_ponderado"].mean(),
        "melhores_params": "-",
    }


def run(niveis: int = 5) -> None:
    if not config.CLEAN_DATASET.exists():
        raise FileNotFoundError(
            "dataset_limpo.csv não encontrado. Rode antes: python -m src.preprocess"
        )

    df = pd.read_csv(config.CLEAN_DATASET)
    df = df.dropna(subset=["texto_limpo", config.LABEL_COL]).reset_index(drop=True)
    X = df["texto_limpo"].astype(str)
    y = df[config.LABEL_COL].astype(str).str.lower()

    # Análise complementar em 3 níveis: colapsa a escala e usa nomes de arquivo
    # com sufixo, SEM sobrescrever os artefatos canônicos de 5 níveis.
    if niveis == 3:
        y = data_utils.collapse_to_3(y)
        ordem_rotulos = config.DIFFICULTY_LABELS_3
        sufixo = "_3niveis"
    else:
        ordem_rotulos = config.DIFFICULTY_LABELS
        sufixo = ""
    # Ambas as granularidades salvam seus artefatos, com sufixo distinto, sem se
    # sobrescreverem (o produto/predict usa 3 níveis; o relatório também analisa 5).
    salvar_modelos = True
    best_model_path = config.MODELS_DIR / f"best_ml_model{sufixo}.joblib"

    cv = data_utils.make_cv(y)
    print(f"Granularidade: {niveis} níveis")
    print(f"Base rotulada: {len(df)} exemplos | {y.nunique()} classes")
    print("Distribuição:")
    print(y.value_counts().to_string())
    print(f"Validação cruzada: {cv.get_n_splits()} folds\n")

    linhas = [_baseline_majoritario(X, y, cv)]
    melhor = {"nome": None, "f1": -1.0, "estimador": None, "pred": None}
    for nome, (estimador, grade) in get_models().items():
        pipe = Pipeline([("tfidf", data_utils.build_vectorizer()), ("clf", estimador)])
        busca = GridSearchCV(
            pipe, grade, scoring=SCORING, refit="f1_macro", cv=cv, n_jobs=-1,
        )
        busca.fit(X, y)

        i = busca.best_index_
        res = busca.cv_results_
        registro = {
            "modelo": nome,
            "acuracia": res["mean_test_acuracia"][i],
            "acuracia_std": res["std_test_acuracia"][i],
            "f1_macro": res["mean_test_f1_macro"][i],
            "f1_macro_std": res["std_test_f1_macro"][i],
            "f1_ponderado": res["mean_test_f1_ponderado"][i],
            "melhores_params": busca.best_params_,
        }
        linhas.append(registro)

        # Predições out-of-fold (cada exemplo previsto pelo fold em que foi teste)
        # para um relatório por classe e uma matriz de confusão honestos.
        pred_oof = cross_val_predict(busca.best_estimator_, X, y, cv=cv)

        print(f"=== {nome} ===")
        print(f"melhores params : {busca.best_params_}")
        print(
            f"acurácia: {registro['acuracia']:.3f} ± {registro['acuracia_std']:.3f} | "
            f"F1-macro: {registro['f1_macro']:.3f} ± {registro['f1_macro_std']:.3f} | "
            f"F1-pond.: {registro['f1_ponderado']:.3f}"
        )
        print(classification_report(y, pred_oof, zero_division=0))

        if salvar_modelos:
            joblib.dump(busca.best_estimator_, config.MODELS_DIR / f"ml_{_slug(nome)}{sufixo}.joblib")
        if registro["f1_macro"] > melhor["f1"]:
            melhor = {
                "nome": nome,
                "f1": registro["f1_macro"],
                "estimador": busca.best_estimator_,
                "pred": pred_oof,
            }

    metrics = pd.DataFrame(linhas).sort_values("f1_macro", ascending=False)
    metrics_csv = config.MODELS_DIR / f"ml_metrics{sufixo}.csv"
    metrics.to_csv(metrics_csv, index=False)
    if salvar_modelos:
        # GridSearchCV(refit="f1_macro") já reajustou o melhor estimador em toda
        # a base; salvamos esse Pipeline (TF-IDF + classificador) autossuficiente.
        joblib.dump(melhor["estimador"], best_model_path)

    # Matriz de confusão out-of-fold do melhor modelo (ordem canônica dos níveis).
    rotulos = [c for c in ordem_rotulos if c in set(y)]
    cm = confusion_matrix(y, melhor["pred"], labels=rotulos)
    cm_df = pd.DataFrame(cm, index=rotulos, columns=rotulos)
    cm_df.to_csv(config.MODELS_DIR / f"matriz_confusao{sufixo}.csv")

    print("=== Resumo (validação cruzada, ordenado por F1-macro) ===")
    print(metrics[["modelo", "acuracia", "f1_macro", "f1_ponderado"]].to_string(index=False))
    print(f"\nMatriz de confusão — {melhor['nome']} (linhas=verdadeiro, colunas=previsto):")
    print(cm_df.to_string())
    print(f"\nMelhor modelo: {melhor['nome']} (F1-macro = {melhor['f1']:.3f})")
    print(f"Métricas salvas em: {metrics_csv}")
    if salvar_modelos:
        print(f"Pipeline salvo em : {best_model_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Etapa 3 - Modelos tradicionais de ML")
    parser.add_argument(
        "--niveis", type=int, choices=[5, 3], default=5,
        help="granularidade da escala de dificuldade (5 = padrão; 3 = análise "
             "complementar colapsada, não sobrescreve os artefatos de 5 níveis)",
    )
    args = parser.parse_args()
    run(niveis=args.niveis)


if __name__ == "__main__":
    main()
