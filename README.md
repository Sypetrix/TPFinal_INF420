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

> Os dados originais ficam em **`arquivos/`**, organizada por **fonte** (uma
> subpasta cada: `INF110/`, `Neps/`, `SPOJ/`, `OBI/`). Cada fonte segue um de
> dois formatos: **`feedbacks`** — enunciados em `txt/`/`txt_with_example/` +
> avaliações dos alunos em `feedbacks_<ano>.json` (ex.: `INF110`) — ou
> **`judge_json`** — um único JSON com a dificuldade já dada pelo juiz (ex.:
> `Neps`). A fonte ativa é definida por `DATASET` no `.env` (padrão `INF110`) e a
> **Etapa 1** (`src.ingest`) detecta o formato e consolida tudo em
> `data/raw/<DATASET>/questoes.csv` automaticamente. *Obs.: `SPOJ` e `OBI` não
> têm dificuldade rotulada (ficam fora do classificador), mas são consolidados e
> entram no **recomendador por conteúdo** — ver §5.*

## 2. Estrutura do projeto

```
TP_Final_INF420/
├── arquivos/           # dados originais, uma subpasta por fonte
│   ├── INF110/             # formato 'feedbacks' (DATASET=INF110, padrão)
│   │   ├── txt/                # enunciado puro (exercise_<id>.txt)
│   │   ├── txt_with_example/   # enunciado + casos de exemplo
│   │   ├── tex/                # versão LaTeX (não usada no ML)
│   │   └── feedbacks_*.json    # avaliações dos alunos (nota 1-5 por questão)
│   ├── Neps/               # formato 'judge_json' (dificuldade dada pelo juiz)
│   │   └── Neps_Academy_complete.json
│   ├── SPOJ/               # 'judge_json' sem rótulo — só no recomendador
│   │   └── SPOJ-BR_complete.json
│   └── OBI/                # 'judge_json' sem rótulo — só no recomendador
│       └── OBI_complete.json
├── data/                   # saídas separadas por fonte: <DATASET>/
│   ├── raw/<DATASET>/      # questoes.csv consolidado (gerado pela Etapa 1)
│   └── processed/<DATASET>/# saídas do pré-processamento (geradas)
├── models/<DATASET>/   # vetorizador e modelos treinados (gerados)
├── notebook/
│   └── exploracao.ipynb  # Análise Exploratória (EDA)
├── src/
│   ├── config.py       # configuração central (.env, caminhos, colunas)
│   ├── data_utils.py   # carregamento, split treino/teste e limpeza de texto
│   ├── gemini_client.py# cliente do Google Gemini
│   ├── ingest.py       # Etapa 1: arquivos/ -> data/raw/questoes.csv
│   ├── preprocess.py   # Etapa 2: limpeza + TF-IDF
│   ├── train_ml.py     # Etapa 3: LogReg, KNN, SVM, RF (validação cruzada)
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
| 1. Ingestão | Lê `arquivos/INF110/` (enunciados + feedbacks) → `questoes.csv` | `python -m src.ingest` |
| 1b. EDA | Distribuição das classes, tamanho dos enunciados | abrir `notebook/exploracao.ipynb` |
| 2. Pré-processamento | Limpeza de texto + vetorização TF-IDF | `python -m src.preprocess` |
| 3. Modelos tradicionais | LogReg, KNN, SVM, RF — validação cruzada + tuning | `python -m src.train_ml` |
| 4. Baseline LLM | Gemini classifica direto (zero-/few-shot) | `python -m src.llm_baseline --n 30 --few-shot` |
| 5. LLM extrator de features | Gemini identifica conceitos (DP, grafos…) | `python -m src.llm_features` |
| 6. LLM explicador | Gera justificativas das classificações | `python -m src.llm_explain --n 5` |
| 7. Avaliação final | Compara ML puro vs LLM vs ML+features LLM | `python -m src.evaluate --n 40` |
| ➕ Recomendação | Sugere exercícios ao aluno (catálogo multi-fonte, §5) | `python -m src.recommend` |

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

> **Metodologia de avaliação (Etapa 3).** Como a base rotulada é pequena, a
> comparação entre os modelos usa **validação cruzada estratificada** (não um
> único holdout), reportando **acurácia, F1-macro e F1-ponderado** como
> média ± desvio entre os folds, além de uma **matriz de confusão** e um
> relatório por classe com predições *out-of-fold*. O **TF-IDF é reajustado
> dentro de cada fold** (via `Pipeline`), nunca vendo o conjunto de teste —
> evitando vazamento de informação. Cada família passa por uma pequena busca de
> hiperparâmetros (`GridSearchCV`), e o melhor modelo (por F1-macro) é reajustado
> em toda a base e salvo como um `Pipeline` autossuficiente (TF-IDF + classificador).
> O nº de folds (`CV_FOLDS`, padrão 5) é reduzido automaticamente quando uma
> classe é muito rara (ex.: `muito_dificil` com 1 questão). Inclui um *baseline*
> de classe majoritária como piso de comparação.

> **Análise de granularidade e figuras (relatório).** `python -m src.train_ml
> --niveis 3` repete a avaliação colapsando a escala para 3 níveis
> (fácil/médio/difícil) — sem sobrescrever os artefatos de 5 níveis — gerando
> `ml_metrics_3niveis.csv` e `matriz_confusao_3niveis.csv`. Depois,
> `python -m src.figuras` gera as figuras do artigo em `figuras/` (mapa de calor
> da matriz de confusão e comparação de F1-macro 5 × 3 níveis). O relatório final
> está em `main.tex` (classe `webmedia`, bibliografia em `referencias.bib`); para
> compilar, é preciso o `webmedia.cls` (fornecido pelo professor em `sample/`, que
> não é versionado).

## 4. Como os dados são consolidados (Etapa 1)

A fonte ativa é escolhida por `DATASET` no `.env`; a Etapa 1 detecta o formato
(ou use `DATASET_FORMAT` para forçar) e gera `data/raw/<DATASET>/questoes.csv`.
Em ambos os formatos, são **5 níveis** na escala canônica:

| Nota | Rótulo | Neps (juiz) |
|------|--------|-------------|
| 1 | `muito_facil`   | Super Fácil   |
| 2 | `facil`         | Fácil         |
| 3 | `medio`         | Médio         |
| 4 | `dificil`       | Difícil       |
| 5 | `muito_dificil` | Super Difícil |

**Formato `feedbacks` (ex.: INF110).** Cada questão tem um `id` (do nome
`exercise_<id>.txt`); o texto vem de `txt_with_example/` (ou `txt/`, conforme
`USE_EXAMPLES`) e o **rótulo** é derivado das notas 1–5 dos alunos em
`feedbacks_*.json`. A nota da questão é a **média aritmética das avaliações
arredondada ao inteiro mais próximo** (`LABEL_STRATEGY=media`, padrão; `moda`
também disponível) — a média evita classificações falsas em casos bimodais (ex.:
21 alunos votam 1 e 20 votam 5 → média ≈ 3 → `medio`). Das 138 questões do
INF110, **80 têm avaliações** (treino/avaliação supervisionados) e as **58 sem
avaliação** entram apenas no recomendador.

**Formato `judge_json` (ex.: Neps Academy).** A fonte é um único JSON em que cada
questão já traz a dificuldade **atribuída pelo juiz** (`metadata.Difficulty`, em
português); o texto vem de `Problem_Description` + `Input`/`Output` (+ `Test_Case`
quando `USE_EXAMPLES`). Não há notas de alunos para agregar — o rótulo do juiz é
mapeado direto pela coluna acima. Das **1448 questões do Neps**, 1333 têm
dificuldade rotulada e 115 ficam sem rótulo (só no recomendador).

A base consolidada `data/raw/<DATASET>/questoes.csv` tem as colunas:

| Coluna (.env) | Padrão | Descrição |
|---------------|--------|-----------|
| `TEXT_COL`  | `enunciado`   | texto do enunciado da questão |
| `LABEL_COL` | `dificuldade` | um dos 5 níveis (`muito_facil`…`muito_dificil`) ou vazio |
| `ID_COL`    | `id`          | identificador da questão |
| —           | `n_avaliacoes` | nº de avaliações de alunos (`feedbacks`); `1`/`0` no `judge_json` |
| —           | `media_dificuldade` / `moda_dificuldade` | estatísticas das notas (no `judge_json`, o próprio nível) |

> **Não apague a pasta `arquivos/`** — ela é a fonte dos dados. Tudo em
> `data/raw/` e `data/processed/` é **gerado a partir dela** e não é versionado
> (`.gitignore`); se apagar `arquivos/`, não será possível regenerar a base nem
> trocar `USE_EXAMPLES` / `LABEL_STRATEGY`.

> **Adicionar uma nova fonte:** crie `arquivos/<Nome>/`, aponte `DATASET=<Nome>`
> no `.env` e rode a Etapa 1 (`python -m src.ingest`). Se for formato `feedbacks`,
> use a estrutura `txt/`, `txt_with_example/` e `feedbacks_*.json`; se for
> `judge_json`, basta um JSON com `ID`, `Problem_Description` e
> `metadata.Difficulty` por questão. O formato é detectado sozinho (ou force em
> `DATASET_FORMAT`). Ex.: `DATASET=Neps python -m src.ingest`.

## 5. Recomendador (catálogo multi-fonte)

O recomendador (`src.recommend`) é **baseado em conteúdo**: representa cada
questão por seu vetor TF-IDF, modela o aluno como o centroide das questões que
ele já resolveu e sugere as não resolvidas mais similares (similaridade do
cosseno), opcionalmente filtrando pelo **próximo nível** de dificuldade.

Como ele não depende de rótulo, **combina várias fontes num só catálogo** —
inclusive as **não rotuladas** (`SPOJ`, `OBI`), que não servem ao classificador
mas são candidatas válidas aqui. As fontes vêm de `RECOMMENDER_SOURCES` no `.env`
(separadas por vírgula; vazio = só a fonte ativa). O padrão já combina as quatro:

```bash
RECOMMENDER_SOURCES=INF110,Neps,SPOJ,OBI    # no .env
python -m src.recommend
```

Cada fonte listada precisa ter sido consolidada antes
(`DATASET=<fonte> python -m src.ingest`). O filtro por nível só se aplica às
questões rotuladas; a recomendação por conteúdo puro percorre todo o catálogo
(e aí `SPOJ`/`OBI` também aparecem). O TF-IDF do recomendador é ajustado sobre o
catálogo combinado, independente do vetorizador do classificador.
