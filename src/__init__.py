"""Pacote do TP Final de INF420.

Classificação automática da dificuldade de questões de programação
(fácil/médio/difícil) com Aprendizado de Máquina e LLM (provedor configurável,
padrão Groq/Llama), além de um recomendador de exercícios por conceitos.

Módulos (na ordem do fluxo de trabalho):
    config            -> configuração central (.env, caminhos, colunas, provedor)
    data_utils        -> carregamento de dados e limpeza de texto
    llm_client        -> cliente modular do LLM (Groq/DeepSeek, padrão OpenAI)
    ingest            -> Etapa 1: lê arquivos/ -> questoes.csv
    preprocess        -> Etapa 2: limpeza + vetorização TF-IDF
    llm_concepts      -> extração de conceitos via LLM (feature de ML + similaridade)
    train_ml          -> Etapa 3: LogReg, KNN, SVM, Random Forest
    llm_baseline      -> baseline de classificação via LLM (comparação, base de treino)
    evaluate          -> métricas e comparação ML puro vs LLM vs ML+conceitos
    predict_difficulty-> aplica o modelo às questões de arquivos/avaliar (dificuldade+conceitos)
    recommend         -> recomendação por conceitos + dificuldade
    llm_explain       -> explica a recomendação (LLM, sob demanda)
"""

__version__ = "0.1.0"
