"""Etapa 3 — Modelos tradicionais de Aprendizado de Máquina.

Treina e compara quatro famílias de modelos sobre as features TF-IDF:
Regressão Logística, KNN, SVM (linear) e Random Forest. Reporta acurácia e
F1-score macro, salva cada modelo e elege o melhor (por F1 macro).

Pré-requisito: rodar antes `python -m src.preprocess`.

Uso:
    python -m src.train_ml
"""
from __future__ import annotations

import argparse

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC

from . import config, data_utils


def get_models() -> dict:
    """Dicionário {nome: estimador} com as quatro famílias de modelos."""
    seed = config.RANDOM_SEED
    return {
        "Regressao Logistica": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=seed
        ),
        "KNN": KNeighborsClassifier(n_neighbors=5, metric="cosine"),
        "SVM (linear)": LinearSVC(class_weight="balanced", random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, class_weight="balanced",
            random_state=seed, n_jobs=-1,
        ),
    }


def _slug(name: str) -> str:
    return (
        name.lower()
        .replace(" ", "_")
        .replace("(", "")
        .replace(")", "")
    )


def run() -> None:
    if not config.CLEAN_DATASET.exists():
        raise FileNotFoundError(
            "dataset_limpo.csv não encontrado. Rode antes: python -m src.preprocess"
        )

    vectorizer = joblib.load(config.TFIDF_VECTORIZER)
    df = pd.read_csv(config.CLEAN_DATASET)
    df = df.dropna(subset=["texto_limpo", config.LABEL_COL]).reset_index(drop=True)

    df_train, df_test = data_utils.split_train_test(df)
    X_train = vectorizer.transform(df_train["texto_limpo"].astype(str))
    X_test = vectorizer.transform(df_test["texto_limpo"].astype(str))
    y_train = df_train[config.LABEL_COL]
    y_test = df_test[config.LABEL_COL]

    print(f"Treino: {X_train.shape[0]} | Teste: {X_test.shape[0]}\n")

    linhas = []
    melhor = {"nome": None, "f1": -1.0, "modelo": None}
    for nome, modelo in get_models().items():
        modelo.fit(X_train, y_train)
        pred = modelo.predict(X_test)
        acc = accuracy_score(y_test, pred)
        f1 = f1_score(y_test, pred, average="macro")
        linhas.append({"modelo": nome, "acuracia": acc, "f1_macro": f1})

        print(f"=== {nome} ===")
        print(classification_report(y_test, pred, zero_division=0))

        joblib.dump(modelo, config.MODELS_DIR / f"ml_{_slug(nome)}.joblib")
        if f1 > melhor["f1"]:
            melhor = {"nome": nome, "f1": f1, "modelo": modelo}

    metrics = pd.DataFrame(linhas).sort_values("f1_macro", ascending=False)
    metrics.to_csv(config.MODELS_DIR / "ml_metrics.csv", index=False)
    joblib.dump(melhor["modelo"], config.BEST_ML_MODEL)

    print("=== Resumo (ordenado por F1 macro) ===")
    print(metrics.to_string(index=False))
    print(f"\nMelhor modelo: {melhor['nome']} (F1 macro = {melhor['f1']:.3f})")
    print(f"Salvo em: {config.BEST_ML_MODEL}")


def main() -> None:
    argparse.ArgumentParser(description="Etapa 3 - Modelos tradicionais de ML").parse_args()
    run()


if __name__ == "__main__":
    main()
