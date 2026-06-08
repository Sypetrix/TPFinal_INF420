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

# Arquivo de dados bruto dentro de data/raw/.
# Se vazio, o data_utils tenta detectar automaticamente o primeiro .csv.
RAW_DATA_FILE = os.getenv("RAW_DATA_FILE", "").strip()

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

# Rótulos canônicos de dificuldade (ordem usada em relatórios/gráficos)
DIFFICULTY_LABELS = ["facil", "medio", "dificil"]

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
