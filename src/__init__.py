"""Pacote do TP Final de INF420.

Classificação automática da dificuldade de questões de programação
(fácil/médio/difícil) com Aprendizado de Máquina e LLMs (Groq/Llama),
além de um recomendador personalizado de exercícios.

Módulos (na ordem do fluxo de trabalho):
    config        -> configuração central (.env, caminhos, colunas)
    data_utils    -> carregamento de dados e limpeza de texto
    llm_client    -> cliente fino para a API da Groq (Llama)
    ingest        -> Etapa 1: lê arquivos/ (enunciados + feedbacks) -> questoes.csv
    preprocess    -> Etapa 2: limpeza + vetorização TF-IDF
    train_ml      -> Etapa 3: LogReg, KNN, SVM, Random Forest
    llm_baseline  -> Etapa 4: classificação direta via LLM (Groq)
    llm_features  -> Etapa 5: LLM como extrator de conceitos
    llm_explain   -> Etapa 6: LLM como gerador de explicações
    evaluate      -> Etapa 7: comparação ML puro vs LLM vs ML+LLM
    predict       -> Inferência: prevê dificuldade + tópico + recomenda (questões novas)
    recommend     -> recomendação personalizada de exercícios
"""

__version__ = "0.1.0"
