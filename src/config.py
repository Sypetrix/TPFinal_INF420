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
# Google Gemini (Google AI Studio)
# ----------------------------------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

# ----------------------------------------------------------------------------
# Caminhos
# ----------------------------------------------------------------------------
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT_DIR / "models"

# Dados de origem entregues pelo professor (enunciados + avaliações dos alunos).
ARQUIVOS_DIR = ROOT_DIR / os.getenv("ARQUIVOS_DIR", "arquivos").strip()
TXT_DIR = ARQUIVOS_DIR / "txt"                       # enunciados puros
TXT_EXEMPLOS_DIR = ARQUIVOS_DIR / "txt_with_example"  # enunciados + casos de exemplo
TEX_DIR = ARQUIVOS_DIR / "tex"                        # versão LaTeX (não usada no ML)

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

# ----------------------------------------------------------------------------
# Reprodutibilidade dos experimentos
# ----------------------------------------------------------------------------
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))

# Garante que as pastas de saída existam
for _d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


def require_api_key() -> str:
    """Retorna a chave do Gemini ou levanta um erro amigável se ausente."""
    if not GEMINI_API_KEY or GEMINI_API_KEY == "sua_chave_aqui":
        raise RuntimeError(
            "GEMINI_API_KEY não configurada. Edite o arquivo .env na raiz do "
            "projeto e coloque sua chave do Google AI Studio em "
            "GEMINI_API_KEY (https://aistudio.google.com/app/apikey)."
        )
    return GEMINI_API_KEY
