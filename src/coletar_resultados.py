"""Coletor de resultados — consolida tudo num .txt legível para o banner.

Varre os artefatos gerados pelas etapas (models/<fonte>/*.csv e
data/processed/<fonte>/*.csv) e escreve um resumo único, fácil de conferir e de
copiar para o banner, em ``resultados/resumo_banner.txt``.

NÃO refaz nenhum experimento nem chama a API — apenas LÊ os CSVs que as etapas
já salvaram (ml_metrics, comparacao_final, matrizes de confusão, etc.). Rode
depois de treinar/avaliar as bases.

Uso:
    python -m src.coletar_resultados                  # autodetecta as bases em models/
    python -m src.coletar_resultados --fontes Neps INF110
    python -m src.coletar_resultados --saida resultados/resumo_banner.txt
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"


def _fmt_metrics(csv: Path) -> str:
    """Tabela de métricas (ml_metrics*.csv) em texto, ordenada por F1-macro."""
    if not csv.exists():
        return "  (ainda não gerado — rode src.train_ml)\n"
    df = pd.read_csv(csv)
    # A coluna de rótulo é "modelo" (ml_metrics) ou "abordagem" (comparacao_final).
    rotcol = "modelo" if "modelo" in df.columns else ("abordagem" if "abordagem" in df.columns else df.columns[0])
    df = df.sort_values("f1_macro", ascending=False)
    linhas = []
    larg = max([len(str(m)) for m in df[rotcol]] + [len(rotcol)])
    cab = f"  {rotcol:<{larg}}  {'acuracia':>9}  {'f1_macro':>9}  {'f1_pond.':>9}"
    linhas.append(cab)
    linhas.append("  " + "-" * (len(cab) - 2))
    for _, r in df.iterrows():
        linhas.append(
            f"  {str(r[rotcol]):<{larg}}  "
            f"{r.get('acuracia', float('nan')):>9.3f}  "
            f"{r.get('f1_macro', float('nan')):>9.3f}  "
            f"{r.get('f1_ponderado', float('nan')):>9.3f}"
        )
    melhor = df.iloc[0]
    linhas.append(f"\n  >> Melhor: {melhor[rotcol]} "
                  f"(F1-macro={melhor['f1_macro']:.3f}, acurácia={melhor['acuracia']:.3f})")
    return "\n".join(linhas) + "\n"


def _fmt_csv_simples(csv: Path, titulo_idx: str = "") -> str:
    if not csv.exists():
        return "  (ainda não gerado)\n"
    df = pd.read_csv(csv, index_col=0 if titulo_idx else None)
    return "  " + df.to_string().replace("\n", "\n  ") + "\n"


def _distribuicao(fonte: str) -> str:
    csv = PROCESSED / fonte / "dataset_limpo.csv"
    if not csv.exists():
        return "  (dataset_limpo.csv não encontrado — rode src.preprocess)\n"
    df = pd.read_csv(csv)
    col = "dificuldade" if "dificuldade" in df.columns else None
    if col is None:
        return f"  total de exemplos: {len(df)}\n"
    vc = df[col].dropna().value_counts()
    linhas = [f"  total de exemplos rotulados: {int(vc.sum())}"]
    for nivel, n in vc.items():
        linhas.append(f"    {nivel:<16} {int(n)}")
    return "\n".join(linhas) + "\n"


def _bloco_fonte(fonte: str) -> str:
    md = MODELS / fonte
    out = []
    out.append("=" * 72)
    out.append(f"FONTE: {fonte}")
    out.append("=" * 72)

    out.append("\n[ Distribuição das classes ]")
    out.append(_distribuicao(fonte))

    out.append("[ Modelos de ML — 5 níveis (validação cruzada) ]")
    out.append(_fmt_metrics(md / "ml_metrics.csv"))

    out.append("[ Modelos de ML — 3 níveis (fácil/médio/difícil) ]")
    out.append(_fmt_metrics(md / "ml_metrics_3niveis.csv"))

    out.append("[ Matriz de confusão — melhor modelo, 5 níveis ]")
    out.append(_fmt_csv_simples(md / "matriz_confusao.csv", titulo_idx="x"))

    out.append("[ Matriz de confusão — melhor modelo, 3 níveis ]")
    out.append(_fmt_csv_simples(md / "matriz_confusao_3niveis.csv", titulo_idx="x"))

    out.append("[ Comparação final — ML puro x LLM puro x ML+conceitos ]")
    comp = md / "comparacao_final.csv"
    if comp.exists():
        out.append(_fmt_metrics(comp))
    else:
        out.append("  (ainda não gerado — rode src.evaluate)\n")

    return "\n".join(out) + "\n"


def _autodetect() -> list[str]:
    if not MODELS.exists():
        return []
    return sorted(
        d.name for d in MODELS.iterdir()
        if d.is_dir() and (d / "ml_metrics.csv").exists()
    )


def run(fontes: list[str] | None, saida: Path) -> None:
    fontes = fontes or _autodetect()
    if not fontes:
        raise SystemExit(
            "Nenhuma base com resultados encontrada em models/. "
            "Rode antes: python -m src.preprocess && python -m src.train_ml"
        )
    saida.parent.mkdir(parents=True, exist_ok=True)
    partes = [
        "#" * 72,
        "#  RESUMO DE RESULTADOS — para conferência e montagem do banner",
        f"#  gerado em: {datetime.now():%Y-%m-%d %H:%M:%S}",
        f"#  bases incluídas: {', '.join(fontes)}",
        "#" * 72,
        "",
        "Observação: estes números são LIDOS dos CSVs que cada etapa salvou em",
        "models/<fonte>/ e data/processed/<fonte>/. Para atualizar, re-rode as",
        "etapas (ver COMANDOS.md ou rodar_tudo.ps1) e chame este script de novo.",
        "",
    ]
    for fonte in fontes:
        partes.append(_bloco_fonte(fonte))

    saida.write_text("\n".join(partes), encoding="utf-8")
    print(f"Resumo salvo em: {saida}")
    print(f"Bases incluídas: {', '.join(fontes)}")


def main() -> None:
    p = argparse.ArgumentParser(description="Consolida os resultados num .txt para o banner")
    p.add_argument("--fontes", nargs="*", default=None,
                   help="bases a incluir (padrão: autodetecta as que têm ml_metrics.csv)")
    p.add_argument("--saida", type=Path, default=ROOT / "resultados" / "resumo_banner.txt",
                   help="arquivo .txt de saída")
    args = p.parse_args()
    run(args.fontes, args.saida)


if __name__ == "__main__":
    main()
