"""Explicador da recomendação (LLM, sob demanda).

Papel 3 da LLM no projeto: quando o sistema recomenda uma questão, gera uma
justificativa em linguagem natural de **por que aquela questão foi escolhida**
(ex.: "reforça grafos, conceito da questão avaliada, num nível um pouco acima").
Roda apenas no momento da recomendação (uma chamada por recomendação), não em
toda a base — custo desprezível.

Pré-requisito: chave do provedor de LLM no .env (LLM_PROVIDER).

Uso (normalmente via ``src.predict_difficulty --explicar``):
    python -m src.llm_explain   # demonstração com um exemplo sintético
"""
from __future__ import annotations

import argparse

from . import llm_client

SYSTEM = (
    "Você é um tutor de programação competitiva. Explique de forma curta e "
    "didática por que uma questão é uma boa recomendação para quem acabou de "
    "estudar outra, citando conceitos em comum e a progressão de dificuldade."
)


def explicar_recomendacao(
    enunciado_avaliado: str,
    enunciado_recomendado: str,
    conceitos_comuns: list[str] | None = None,
    relacao_dificuldade: str = "no mesmo nível",
) -> str:
    """Justifica, em 1-2 frases, por que a questão recomendada faz sentido."""
    conceitos = ", ".join(conceitos_comuns) if conceitos_comuns else "conteúdo similar"
    prompt = (
        "Uma pessoa está estudando esta questão:\n"
        f'"{str(enunciado_avaliado)[:1200]}"\n\n'
        "Você vai recomendar a ela esta outra questão:\n"
        f'"{str(enunciado_recomendado)[:1200]}"\n\n'
        f"Conceitos em comum: {conceitos}.\n"
        f"Relação de dificuldade da recomendada: {relacao_dificuldade}.\n\n"
        "Explique em 1 a 2 frases, em português, por que esta é uma boa "
        "recomendação — cite os conceitos reforçados e a progressão de "
        "dificuldade. Seja direto, sem repetir os enunciados."
    )
    return llm_client.generate(prompt, SYSTEM, temperature=0.3)


def run() -> None:
    """Demonstração rápida com um exemplo sintético."""
    avaliada = ("Dado um grafo não-direcionado, encontre o menor caminho entre "
                "dois vértices usando busca em largura (BFS).")
    recomendada = ("Em um labirinto representado por uma grade, encontre o menor "
                   "número de passos do início até a saída.")
    print("Explicação da recomendação:")
    print(explicar_recomendacao(avaliada, recomendada, ["grafos", "busca_binaria"],
                                "um nível acima"))


def main() -> None:
    argparse.ArgumentParser(description="Explicador da recomendação (LLM, sob demanda)").parse_args()
    run()


if __name__ == "__main__":
    main()
