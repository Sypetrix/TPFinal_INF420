# Classificação Automática da Dificuldade de Questões de Programação

Trabalho final de **INF420 (Inteligência Artificial) — UFV**.

Sistema que **classifica a dificuldade** de questões de programação (foco em
maratonas) a partir do enunciado, em **5 níveis** (muito fácil, fácil, médio,
difícil, muito difícil), combinando **Aprendizado de Máquina tradicional** com
**LLMs (Google Gemini Flash)**, e que **recomenda novos exercícios** de forma
personalizada com base no histórico do aluno.

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

> Os dados originais ficam em **`arquivos/`** (entregues pelo professor):
> enunciados em `arquivos/txt/` e `arquivos/txt_with_example/` e as avaliações
> dos alunos em `arquivos/feedbacks_<ano>.json`. A **Etapa 1** (`src.ingest`)
> consolida tudo em `data/raw/questoes.csv` automaticamente.

## 2. Estrutura do projeto

```
TP_Final_INF420/
├── arquivos/           # dados originais do professor
│   ├── txt/                # enunciado puro (exercise_<id>.txt)
│   ├── txt_with_example/   # enunciado + casos de exemplo
│   ├── tex/                # versão LaTeX (não usada no ML)
│   └── feedbacks_*.json    # avaliações dos alunos (nota 1-5 por questão)
├── data/
│   ├── raw/            # questoes.csv consolidado (gerado pela Etapa 1)
│   └── processed/      # saídas do pré-processamento (geradas)
├── models/             # vetorizador e modelos treinados (gerados)
├── notebook/
│   └── exploracao.ipynb  # Análise Exploratória (EDA)
├── src/
│   ├── config.py       # configuração central (.env, caminhos, colunas)
│   ├── data_utils.py   # carregamento, split treino/teste e limpeza de texto
│   ├── gemini_client.py# cliente do Google Gemini
│   ├── ingest.py       # Etapa 1: arquivos/ -> data/raw/questoes.csv
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
| 1. Ingestão | Lê `arquivos/` (enunciados + feedbacks) → `questoes.csv` | `python -m src.ingest` |
| 1b. EDA | Distribuição das classes, tamanho dos enunciados | abrir `notebook/exploracao.ipynb` |
| 2. Pré-processamento | Limpeza de texto + vetorização TF-IDF | `python -m src.preprocess` |
| 3. Modelos tradicionais | LogReg, KNN, SVM, Random Forest (F1-score) | `python -m src.train_ml` |
| 4. Baseline LLM | Gemini classifica direto (zero-/few-shot) | `python -m src.llm_baseline --n 30 --few-shot` |
| 5. LLM extrator de features | Gemini identifica conceitos (DP, grafos…) | `python -m src.llm_features` |
| 6. LLM explicador | Gera justificativas das classificações | `python -m src.llm_explain --n 5` |
| 7. Avaliação final | Compara ML puro vs LLM vs ML+features LLM | `python -m src.evaluate --n 40` |
| ➕ Recomendação | Sugere próximos exercícios ao aluno | `python -m src.recommend` |

> **Etapa 1 automática:** rodar a Etapa 2 (`src.preprocess`) já dispara a Etapa 1
> sozinha se `data/raw/questoes.csv` ainda não existir — então o mínimo para
> treinar os modelos de ML é `python -m src.preprocess` seguido de
> `python -m src.train_ml`.

> **Dica de custo de API:** as etapas 4, 5, 6 e a parte LLM da 7 chamam o
> Gemini e exigem `GEMINI_API_KEY` no `.env`. Comece com amostras pequenas
> (`--n`). Para rodar a avaliação final **sem** chamar a API, use
> `python -m src.evaluate --no-llm` (compara só ML puro vs ML+features). A camada
> gratuita do Google AI Studio tem limite por minuto — o cliente já refaz
> tentativas, e a Etapa 5 tem retomada (não refaz linhas já processadas).

## 4. Como os dados são consolidados (Etapa 1)

Cada questão tem um `id` (do nome `exercise_<id>.txt`). O texto vem de
`txt_with_example/` (configurável em `USE_EXAMPLES`) e o **rótulo** é derivado das
notas 1–5 que os alunos deram nos `feedbacks_*.json`. São **5 níveis**, casando
diretamente com a escala de avaliação:

| Nota (1–5) | Rótulo |
|------------|--------|
| 1 | `muito_facil` |
| 2 | `facil` |
| 3 | `medio` |
| 4 | `dificil` |
| 5 | `muito_dificil` |

A nota da questão é a **média aritmética das avaliações arredondada ao inteiro
mais próximo** (`LABEL_STRATEGY=media`, padrão; `moda` também disponível). A média
evita classificações falsas em casos bimodais — ex.: 21 alunos votam 1 e 20 votam
5, a média ≈ 3 classifica como `medio`. Das 138 questões, **80 têm avaliações**
(usadas no treino/avaliação supervisionados) e as **58 sem avaliação** entram
apenas no recomendador (similaridade de conteúdo, que não exige rótulo).

A base consolidada `data/raw/questoes.csv` tem as colunas:

| Coluna (.env) | Padrão | Descrição |
|---------------|--------|-----------|
| `TEXT_COL`  | `enunciado`   | texto do enunciado da questão |
| `LABEL_COL` | `dificuldade` | um dos 5 níveis (`muito_facil`…`muito_dificil`) ou vazio |
| `ID_COL`    | `id`          | identificador da questão |
| —           | `n_avaliacoes` | nº de avaliações de alunos |
| —           | `media_dificuldade` / `moda_dificuldade` | estatísticas das notas |

> **Não apague a pasta `arquivos/`** — ela é a fonte dos dados. Tudo em
> `data/raw/` e `data/processed/` é **gerado a partir dela** e não é versionado
> (`.gitignore`); se apagar `arquivos/`, não será possível regenerar a base nem
> trocar `USE_EXAMPLES` / `LABEL_STRATEGY`.
