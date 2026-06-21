"""Configuração central do projeto.

Carrega variáveis do arquivo .env e define caminhos, nomes de colunas e
hiperparâmetros compartilhados por todos os módulos. Importe a partir daqui
em vez de espalhar caminhos/strings pelo código.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# Raiz do projeto (um nível acima de src/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Carrega o .env da raiz do projeto
load_dotenv(ROOT_DIR / ".env")

# ----------------------------------------------------------------------------
# Groq Cloud (modelos Llama) — provedor do LLM
# ----------------------------------------------------------------------------
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
# Default no modelo com a maior cota gratuita diária disponível
# (llama-3.1-8b-instant: ~14.400 requisições/dia). Troque por GROQ_MODEL no .env
# se preferir outro (ver comentário no .env.example).
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant").strip()

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"

# Pasta-container com TODAS as fontes de dados (uma subpasta por fonte:
# INF110, Neps, SPOJ, OBI, ...). Os arquivos do professor ficam em
# arquivos/INF110/.
ARQUIVOS_DIR = ROOT_DIR / os.getenv("ARQUIVOS_DIR", "arquivos").strip()

# Fonte de dados ativa (subpasta de arquivos/). Troque DATASET no .env para
# apontar o pipeline a outro banco de questões (ex.: Neps).
DATASET = os.getenv("DATASET", "INF110").strip()
DATASET_DIR = ARQUIVOS_DIR / DATASET

# Formato da fonte ativa — como a Etapa 1 (src.ingest) lê os dados brutos:
#   auto        -> detecta automaticamente (padrão)
#   feedbacks   -> enunciados em txt/ + feedbacks_*.json (ex.: INF110)
#   judge_json  -> um único JSON com a dificuldade já dada pelo juiz (ex.: Neps)
DATASET_FORMAT = os.getenv("DATASET_FORMAT", "auto").strip().lower()

# Fontes usadas pelo recomendador (lista separada por vírgula). O recomendador é
# baseado em conteúdo (similaridade), então questões SEM rótulo (SPOJ, OBI) também
# servem como candidatas. Vazio -> usa só a fonte ativa (DATASET). Cada fonte
# listada precisa ter sido consolidada antes (data/raw/<fonte>/questoes.csv).
RECOMMENDER_SOURCES = [
    s.strip() for s in os.getenv("RECOMMENDER_SOURCES", "").split(",") if s.strip()
]

TXT_DIR = DATASET_DIR / "txt"                       # enunciados puros
TXT_EXEMPLOS_DIR = DATASET_DIR / "txt_with_example"  # enunciados + casos de exemplo
TEX_DIR = DATASET_DIR / "tex"                        # versão LaTeX (não usada no ML)

# Saídas geradas, separadas por fonte ativa (data/raw/<DATASET>/, etc.), para
# que processar uma fonte (ex.: Neps) não sobrescreva os artefatos de outra
# (ex.: INF110). Todas as etapas usam estes caminhos via config.
RAW_DIR = DATA_DIR / "raw" / DATASET
PROCESSED_DIR = DATA_DIR / "processed" / DATASET
MODELS_DIR = ROOT_DIR / "models" / DATASET


def questoes_csv_for(dataset: str) -> Path:
    """Caminho do questoes.csv consolidado de uma fonte qualquer (não só a ativa).

    Útil ao recomendador, que pode combinar várias fontes num só catálogo.
    """
    return DATA_DIR / "raw" / dataset.strip() / "questoes.csv"

# Usar a versão com exemplos (casos de teste) como texto do enunciado?
USE_EXAMPLES = os.getenv("USE_EXAMPLES", "true").strip().lower() in {"1", "true", "sim", "yes"}

# Como agregar as notas 1-5 dos alunos no rótulo de dificuldade (5 níveis):
#   "media" -> média aritmética arredondada ao inteiro mais próximo (padrão)
#   "moda"  -> nota mais frequente (empate: a mais próxima da média)
# Em ambos, a nota inteira 1-5 vira o nível correspondente (ver DIFFICULTY_LABELS).
LABEL_STRATEGY = os.getenv("LABEL_STRATEGY", "media").strip().lower()

# Arquivo de dados bruto dentro de data/raw/, gerado pela Etapa 1 (src.ingest).
# Se vazio, o data_utils tenta detectar automaticamente o primeiro .csv.
RAW_DATA_FILE = os.getenv("RAW_DATA_FILE", "questoes.csv").strip()
QUESTOES_CSV = RAW_DIR / "questoes.csv"

# Saídas padronizadas (geradas pelos scripts)
CLEAN_DATASET = PROCESSED_DIR / "dataset_limpo.csv"
TFIDF_VECTORIZER = MODELS_DIR / "tfidf_vectorizer.joblib"
TFIDF_MATRIX = MODELS_DIR / "tfidf_matrix.joblib"
BEST_ML_MODEL = MODELS_DIR / "best_ml_model.joblib"
LLM_BASELINE_PREDS = PROCESSED_DIR / "llm_baseline_preds.csv"
LLM_FEATURES = PROCESSED_DIR / "llm_features.csv"

# ----------------------------------------------------------------------------
# Esquema esperado da base (ajuste conforme o dataset do professor)
# ----------------------------------------------------------------------------
TEXT_COL = os.getenv("TEXT_COL", "enunciado").strip()
LABEL_COL = os.getenv("LABEL_COL", "dificuldade").strip()
ID_COL = os.getenv("ID_COL", "id").strip()

# Rótulos canônicos de dificuldade (ordem usada em relatórios/gráficos).
# Cada nível corresponde diretamente a uma nota de 1 a 5 dada pelos alunos.
DIFFICULTY_LABELS = ["muito_facil", "facil", "medio", "dificil", "muito_dificil"]

# Mapeamento nota inteira (1-5) -> rótulo canônico.
NOTA_PARA_ROTULO = {
    1: "muito_facil",
    2: "facil",
    3: "medio",
    4: "dificil",
    5: "muito_dificil",
}

# Inverso: rótulo canônico -> nota inteira (1-5).
ROTULO_PARA_NOTA = {rotulo: nota for nota, rotulo in NOTA_PARA_ROTULO.items()}

# Escala reduzida de 3 níveis (fácil/médio/difícil), usada na análise
# complementar de granularidade. Mapeia os 5 níveis para os 3 e dá a ordem
# canônica dessa escala.
DIFFICULTY_LABELS_3 = ["facil", "medio", "dificil"]
COLLAPSE_5_TO_3 = {
    "muito_facil": "facil",
    "facil": "facil",
    "medio": "medio",
    "dificil": "dificil",
    "muito_dificil": "dificil",
}

# Fontes que já trazem a dificuldade rotulada pelo juiz (formato "judge_json"),
# sem notas de alunos para agregar. Mapeia o rótulo do juiz para a escala
# canônica de 5 níveis. Chaves normalizadas (minúsculas, sem acento) — o casamento
# é feito por ingest._normalizar, então acentuação/caixa do JSON não importam.
NEPS_DIFFICULTY_MAP = {
    "super facil": "muito_facil",
    "facil": "facil",
    "medio": "medio",
    "dificil": "dificil",
    "super dificil": "muito_dificil",
}

# ----------------------------------------------------------------------------
# Vetorização TF-IDF (compartilhada por preprocess, train_ml e evaluate, para
# que todas as etapas usem exatamente a mesma configuração de features)
# ----------------------------------------------------------------------------
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "5000"))
TFIDF_NGRAM_MAX = int(os.getenv("TFIDF_NGRAM_MAX", "2"))
TFIDF_MIN_DF = int(os.getenv("TFIDF_MIN_DF", "2"))

# ----------------------------------------------------------------------------
# Reprodutibilidade dos experimentos
# ----------------------------------------------------------------------------
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

# Nº de folds da validação cruzada usada na avaliação dos modelos (Etapa 3).
# É automaticamente reduzido se a menor classe tiver menos exemplos que isso.
CV_FOLDS = int(os.getenv("CV_FOLDS", "5"))

# Garante que as pastas de saída existam
for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def require_api_key() -> str:
    """Retorna a chave da Groq ou levanta um erro amigável se ausente."""
    if not GROQ_API_KEY or GROQ_API_KEY == "sua_chave_aqui":
        raise RuntimeError(
            "GROQ_API_KEY não configurada. Edite o arquivo .env na raiz do "
            "projeto e coloque sua chave da Groq em GROQ_API_KEY "
            "(crie em https://console.groq.com/keys)."
        )
    return GROQ_API_KEY
