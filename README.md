# Classificação Automática da Dificuldade de Questões de Programação

Trabalho final de **INF420 (Inteligência Artificial) — UFV**.

Sistema que **classifica a dificuldade** de questões de programação (foco em
maratonas) a partir do enunciado, em **5 níveis** (muito fácil, fácil, médio,
difícil, muito difícil), combinando **Aprendizado de Máquina tradicional** com
**LLMs (Groq / Llama)**, e que **recomenda novos exercícios** de forma
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

# 3. Configure a chave da Groq
cp .env.example .env             # depois edite o .env
#   -> coloque sua chave em GROQ_API_KEY (https://console.groq.com/keys)
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
│   ├── config.py            # configuração central (.env, caminhos, provedor LLM)
│   ├── data_utils.py        # carregamento, split treino/teste e limpeza de texto
│   ├── llm_client.py        # cliente LLM modular (Groq/DeepSeek, padrão OpenAI)
│   ├── ingest.py            # Etapa 1: arquivos/ -> data/raw/questoes.csv
│   ├── preprocess.py        # Etapa 2: limpeza + TF-IDF
│   ├── llm_concepts.py      # extração de conceitos via LLM (feature ML + similaridade)
│   ├── train_ml.py          # Etapa 3: LogReg, KNN, SVM, RF (validação cruzada)
│   ├── llm_baseline.py      # baseline de classificação via LLM (comparação, base de treino)
│   ├── evaluate.py          # métricas: ML puro vs LLM vs ML+conceitos
│   ├── predict_difficulty.py# aplica o modelo a arquivos/avaliar (dificuldade ML + conceitos)
│   ├── recommend.py         # recomendação por conceitos + dificuldade
│   └── llm_explain.py       # explica a recomendação (LLM, sob demanda)
├── .env                # chave(s) do provedor de LLM (não versionado)
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
| 4. Conceitos (LLM) | LLM identifica conceitos (DP, grafos…) — feature de ML e similaridade | `python -m src.llm_concepts --n 10` |
| 5. Baseline LLM | LLM classifica direto (zero-/few-shot) — só comparação, base de treino | `python -m src.llm_baseline --n 10 --few-shot` |
| 6. Avaliação final | Compara ML puro vs LLM vs ML+conceitos (acurácia/F1) | `python -m src.evaluate --n 20` (ou `--no-llm`) |
| 🎯 Inferência (questões novas) | Dificuldade (ML) + conceitos (LLM) + recomenda por conceitos/nível (§6) | `python -m src.predict_difficulty --fonte avaliar` |
| ➕ Recomendação | Recomendador por conceitos + dificuldade (catálogo multi-fonte, §5) | `python -m src.recommend` |
| ✎ Explicação | Justifica uma recomendação (sob demanda) | via `--explicar` na inferência |

> ⚠️ Os comandos das etapas 4–7 acima usam amostras **pequenas** de propósito —
> cada item vira uma chamada à API do LLM (Groq), que tem cota gratuita limitada.
> **Aumente o `--n` aos poucos.** Detalhes e solução de problemas de cota em §3.1.

> **Etapa 1 automática:** rodar a Etapa 2 (`src.preprocess`) já dispara a Etapa 1
> sozinha se `data/raw/<DATASET>/questoes.csv` ainda não existir — então o mínimo
> para treinar os modelos de ML é `python -m src.preprocess` seguido de
> `python -m src.train_ml`. **Nada disso usa a API** (ver §3.1).

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

### 3.1 Etapas com LLM: provedor, chave, custo e limites da API

O **provedor de LLM é modular** (`LLM_PROVIDER` no `.env`): **Groq** (padrão,
Llama) ou **DeepSeek** — ambos com API no padrão OpenAI, trocáveis numa linha. As
etapas que usam LLM (`llm_concepts`, `llm_baseline`, `evaluate` sem `--no-llm`, e a
parte LLM da inferência) exigem a chave do provedor ativo no `.env`
(Groq: <https://console.groq.com/keys>). Já a **ingestão, o pré-processamento, o
treino, as figuras e a inferência com `--no-llm` NÃO usam a API** — o resultado do
classificador é reproduzível **offline**. Importante: a **dificuldade das questões
a avaliar é decidida pelo ML**, não pela LLM (a LLM extrai conceitos e explica).

**Comece pequeno** — cada item vira **uma** chamada à API:

```bash
python -m src.llm_concepts  --n 5              # 5 chamadas (tem retomada)
python -m src.llm_baseline  --n 5 --few-shot   # 5 chamadas (comparação)
python -m src.evaluate      --n 10            # ~10 chamadas (abordagem LLM)
python -m src.evaluate      --no-llm          # 0 chamadas (só ML)
```

**A cota gratuita (free tier) da Groq tem limites por minuto e por dia:**
- **RPM / TPM** (requisições / *tokens* por *minuto*): rajadas rápidas ou lotes
  muito grandes estouram → espace as chamadas com `--sleep 4` (disponível em
  `llm_baseline` e `llm_concepts`) e use lotes menores.
- **RPD** (requisições por *dia*): teto **diário** por modelo. O padrão
  `llama-3.1-8b-instant` tem a **maior** cota (~14.400 req/dia); a contagem
  **renova à meia-noite UTC**. Limites oficiais por modelo:
  <https://console.groq.com/docs/rate-limits>.

A extração de conceitos (`llm_concepts`) tem **retomada**: não refaz itens já
gravados no cache, então dá para processar a base aos poucos sem perder progresso.

#### Lotes (prompt packing): menos chamadas para a mesma tarefa

Para gastar menos cota, `llm_concepts`, `llm_baseline`, `evaluate` e a inferência
aceitam `--lote N`, que envia **N enunciados em uma única requisição** (em vez de
uma por item), reduzindo o número de chamadas em ~N×:

```bash
python -m src.llm_concepts  --lote 10                     # base toda em ~14 chamadas
python -m src.llm_baseline --n 40 --few-shot --lote 10   # ~4 chamadas em vez de 40
python -m src.evaluate      --n 40 --lote 10             # baseline LLM em lotes
python -m src.predict_difficulty --fonte avaliar --lote 5 # inferência em lotes
```

A resposta é casada por `id`; se algum item não voltar, ele é reprocessado
sozinho — o resultado tem sempre o mesmo tamanho e a mesma corretude do modo
item-a-item. Comece com `--lote 5`–`10`: lotes grandes de enunciados longos podem
estourar o limite de *tokens* por minuto (TPM).

> **E a "Batch API" oficial?** A Groq também oferece um modo de lote assíncrono
> (~50% do preço). Mas isso é otimização de **custo em dinheiro**, que faz sentido
> no tier **pago**. No **free tier** vocês não pagam em dinheiro — o gargalo é a
> **cota** (RPM/TPM/RPD), e o que a economiza é reduzir o nº de requisições, que é
> exatamente o que o `--lote` (prompt packing) faz. Por isso adotamos essa
> abordagem.

#### "Bati no limite" — o que está acontecendo?

Com o `llama-3.1-8b-instant` a cota **diária** é alta (~14.400 req), então o
estouro costuma ser por **minuto** (RPM/TPM) — recuperável com espera. Causas e
como resolver:

1. **Lote/job grande estourando o TPM.** Lotes grandes de enunciados longos
   passam do teto de *tokens por minuto*. Use `--lote 5` e `--sleep 4`.
2. **Rajada rápida (RPM).** Muitas chamadas em poucos segundos. Espace com
   `--sleep`. O cliente já faz *backoff* automático nesses erros transitórios.
3. **Repetições gastando cota.** *(já tratado)* o cliente **para na hora** ao
   detectar estouro de cota **diária** ou erro de configuração (chave/modelo
   inválidos), em vez de repetir 5× à toa.
4. **Modelo trocado para um menor.** Se usar `llama-3.3-70b-versatile` (mais
   forte, porém cota menor), o teto diário cai — volte ao `8b-instant` para mais
   cota.

Para acompanhar uso, chave e limites: <https://console.groq.com/>.

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

## 6. Inferência: avaliar questões novas (`predict_difficulty`)

Esta é a etapa-fim — o uso do **professor que tem questões novas e quer
avaliá-las**. Para cada questão (ainda **sem rótulo**), `src.predict_difficulty`:

1. **Prevê a dificuldade** com o classificador de ML treinado (em **3 níveis** por
   padrão — `models/<DATASET>/best_ml_model_3niveis.joblib`; use `--niveis 5` para
   5 níveis) — **offline, sem API**. A dificuldade é decidida pelo **ML**, não pela
   LLM;
2. **Identifica os conceitos** (recursão, grafos, programação dinâmica…) com a
   LLM — opcional (`--no-llm`);
3. **Recomenda questões** do banco (catálogo da §5) por **conceitos em comum +
   dificuldade compatível** (mesmo nível ±1 do previsto). Quando não há conceitos,
   cai para similaridade TF-IDF. Quase-duplicatas (a mesma questão) são descartadas
   (`--manter-duplicatas` desliga; `--ignorar-nivel` remove o filtro de nível);
4. *(opcional `--explicar`)* a LLM **justifica cada recomendação**.

Há dois modos de entrada:

```bash
# (a) Uso avaliador: uma pasta arquivos/<Nome>/ com um JSON no formato judge_json
#     (campo metadata.Difficulty em branco = "a avaliar"). O repositório já inclui
#     um exemplo pronto em arquivos/avaliar/ (questões reais do OBI e do SPOJ).
DATASET=avaliar python -m src.ingest                       # consolida a fonte 'avaliar'
python -m src.predict_difficulty --fonte avaliar --no-llm  # offline (ML + recomendação)
python -m src.predict_difficulty --fonte avaliar --lote 5  # completo (+ conceitos)

# (b) Modo teste: amostra questões SEM rótulo já no catálogo, para validar
#     a ferramenta ponta a ponta.
python -m src.predict_difficulty --teste --n 5 --no-llm
```

> **Treine o classificador antes** (uma vez): `python -m src.preprocess` e
> `python -m src.train_ml --niveis 3` geram `best_ml_model_3niveis.joblib`. A
> previsão usa o modelo da **fonte ativa** (`DATASET`): mantenha `DATASET=INF110`
> (ou `Neps`) e use `--fonte avaliar` só para apontar as questões. Dica: **Neps**
> tem ~1.300 questões rotuladas (vs. 80 do INF110), então `DATASET=Neps` tende a
> dar um classificador mais robusto.

> **Recomendação por conceitos:** para o critério de conceitos valer no catálogo,
> extraia-os por fonte uma vez (`python -m src.llm_concepts --fonte Neps --lote 10`,
> idem SPOJ/OBI/INF110). Sem esse cache, a recomendação usa TF-IDF (funciona, mas
> não é o critério ideal). A saída é salva em
> `data/processed/<DATASET>/avaliar_classificado.csv`.

> No modo `--teste`, as questões **não têm rótulo verdadeiro** — demonstra-se o
> **funcionamento** ponta a ponta, não a acurácia (medida na avaliação, `src.evaluate`,
> sobre dados rotulados).
