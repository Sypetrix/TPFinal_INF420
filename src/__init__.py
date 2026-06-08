"""Pacote do TP Final de INF420.

Classificação automática da dificuldade de questões de programação
(fácil/médio/difícil) com Aprendizado de Máquina e LLMs (Google Gemini),
além de um recomendador personalizado de exercícios.

Módulos (na ordem do fluxo de trabalho):
    config        -> configuração central (.env, caminhos, colunas)
    data_utils    -> carregamento de dados e limpeza de texto
    gemini_client -> cliente fino para o Google Gemini
    preprocess    -> Etapa 2: limpeza + vetorização TF-IDF
    train_ml      -> Etapa 3: LogReg, KNN, SVM, Random Forest
    llm_baseline  -> Etapa 4: classificação direta via Gemini
    llm_features  -> Etapa 5: Gemini como extrator de conceitos
    llm_explain   -> Etapa 6: Gemini como gerador de explicações
    evaluate      -> Etapa 7: comparação ML puro vs LLM vs ML+LLM
    recommend     -> recomendação personalizada de exercícios
"""

__version__ = "0.1.0"
