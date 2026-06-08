# Classificação Automática da Dificuldade de Questões de Programação

Trabalho final de **INF420 (Inteligência Artificial) — UFV**.

Sistema que **classifica a dificuldade** (fácil / médio / difícil) de questões
de programação (foco em maratonas) a partir do enunciado, combinando
**Aprendizado de Máquina tradicional** com **LLMs (Google Gemini Flash)**, e que
**recomenda novos exercícios** de forma personalizada com base no histórico do
aluno.

**Autores:** Carlos Eduardo Pereira Oliveira (116233) · Luis Felipe Martins Pereira (112710)

---

## 1. Instalação

```bash
# 1. Crie e ative o ambiente virtual
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Instale as dependências
pip install -r requirements.txt

# 3. Configure a chave do Gemini
cp .env.example .env             # depois edite o .env
#   -> coloque sua chave em GEMINI_API_KEY (https://aistudio.google.com/app/apikey)
```

> A base de dados deve ser colocada em **`data/raw/`** (um `.csv`, `.json` ou
> `.xlsx`). Ajuste os nomes das colunas (`TEXT_COL`, `LABEL_COL`, `ID_COL`) no
> `.env` conforme o seu arquivo.

## 2. Estrutura do projeto

```
TP_Final_INF420/
├── data/
│   ├── raw/            # base original (coloque seu arquivo aqui)
│   └── processed/      # saídas do pré-processamento (geradas)
├── models/             # vetorizador e modelos treinados (gerados)
├── notebook/
│   └── exploracao.ipynb  # Etapa 1: Análise Exploratória (EDA)
├── src/
│   ├── config.py       # configuração central (.env, caminhos, colunas)
│   ├── data_utils.py   # carregamento, split treino/teste e limpeza de texto
│   ├── gemini_client.py# cliente do Google Gemini
│   ├── preprocess.py   # Etapa 2: limpeza + TF-IDF
│   ├── train_ml.py     # Etapa 3: LogReg, KNN, SVM, Random Forest
│   ├── llm_baseline.py # Etapa 4: classificação direta via Gemini
│   ├── llm_features.py # Etapa 5: Gemini como extrator de conceitos
│   ├── llm_explain.py  # Etapa 6: Gemini como explicador
│   ├── evaluate.py     # Etapa 7: comparação ML vs LLM vs ML+LLM
│   └── recommend.py    # recomendação personalizada de exercícios
├── .env                # SUA chave do Gemini (não versionado)
├── .env.example        # modelo de configuração
└── requirements.txt
```

## 3. Fluxo de trabalho (ordem recomendada)

| Etapa | O que faz | Como rodar |
|------:|-----------|------------|
| 1. EDA | Distribuição das classes, tamanho dos enunciados | abrir `notebook/exploracao.ipynb` |
| 2. Pré-processamento | Limpeza de texto + vetorização TF-IDF | `python -m src.preprocess` |
| 3. Modelos tradicionais | LogReg, KNN, SVM, Random Forest (F1-score) | `python -m src.train_ml` |
| 4. Baseline LLM | Gemini classifica direto (zero-/few-shot) | `python -m src.llm_baseline --n 30 --few-shot` |
| 5. LLM extrator de features | Gemini identifica conceitos (DP, grafos…) | `python -m src.llm_features` |
| 6. LLM explicador | Gera justificativas das classificações | `python -m src.llm_explain --n 5` |
| 7. Avaliação final | Compara ML puro vs LLM vs ML+features LLM | `python -m src.evaluate --n 40` |
| ➕ Recomendação | Sugere próximos exercícios ao aluno | `python -m src.recommend` |

> **Dica de custo de API:** as etapas 4, 5, 6 e a parte LLM da 7 chamam o
> Gemini. Comece com amostras pequenas (`--n`). A camada gratuita do Google AI
> Studio tem limite de requisições por minuto — o cliente já refaz tentativas
> automaticamente, e a Etapa 5 tem retomada (não refaz linhas já processadas).

## 4. Esquema esperado da base

Por padrão o código espera as colunas abaixo (renomeáveis no `.env`):

| Coluna (.env) | Padrão | Descrição |
|---------------|--------|-----------|
| `TEXT_COL`  | `enunciado`   | texto do enunciado da questão |
| `LABEL_COL` | `dificuldade` | rótulo: `facil`, `medio` ou `dificil` |
| `ID_COL`    | `id`          | identificador (opcional) |
