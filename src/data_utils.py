"""Utilidades de dados: carregamento, divisão treino/teste e limpeza de texto.

Funções compartilhadas por todas as etapas para garantir consistência
(mesma divisão treino/teste, mesma limpeza de texto).
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

from . import config

# ----------------------------------------------------------------------------
# Carregamento / salvamento
# ----------------------------------------------------------------------------


def load_raw() -> pd.DataFrame:
    """Carrega a base bruta de data/raw/ (.csv, .json ou .xlsx)."""
    path = _resolve_raw_path()
    df = _read_any(path)
    print(f"Base carregada de {path.name}: {df.shape[0]} linhas x {df.shape[1]} colunas")
    return df


def _resolve_raw_path() -> Path:
    if config.RAW_DATA_FILE:
        path = config.RAW_DIR / config.RAW_DATA_FILE
        if not path.exists():
            raise FileNotFoundError(f"Arquivo de dados não encontrado: {path}")
        return path
    candidates = sorted(
        p for p in config.RAW_DIR.glob("*")
        if p.suffix.lower() in {".csv", ".json", ".xlsx"}
    )
    if not candidates:
        raise FileNotFoundError(
            f"Nenhuma base encontrada em {config.RAW_DIR}.\n"
            "Coloque seu arquivo .csv/.json/.xlsx em data/raw/ ou defina "
            "RAW_DATA_FILE no .env."
        )
    return candidates[0]


def _read_any(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".json":
        return pd.read_json(path)
    if suffix == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(f"Formato de arquivo não suportado: {suffix}")


def save_processed(df: pd.DataFrame, name: str) -> Path:
    """Salva um DataFrame em data/processed/<name>."""
    path = config.PROCESSED_DIR / name
    df.to_csv(path, index=False)
    return path


def check_columns(df: pd.DataFrame, columns: list[str]) -> None:
    """Levanta erro claro se alguma coluna esperada não existir na base."""
    faltando = [c for c in columns if c not in df.columns]
    if faltando:
        raise KeyError(
            f"Colunas ausentes na base: {faltando}. "
            f"Colunas disponíveis: {list(df.columns)}. "
            "Ajuste TEXT_COL/LABEL_COL/ID_COL no .env."
        )


def build_vectorizer(
    max_features: int | None = None,
    ngram_max: int | None = None,
    min_df: int | None = None,
) -> TfidfVectorizer:
    """Cria um TF-IDF com os hiperparâmetros padrão do projeto (config).

    Centralizado aqui para que pré-processamento, treino e avaliação usem
    exatamente a mesma configuração de vetorização. Em ``train_ml`` ele entra
    dentro de um ``Pipeline`` e é reajustado a cada fold da validação cruzada,
    evitando vazamento de informação do conjunto de teste.
    """
    return TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES if max_features is None else max_features,
        ngram_range=(1, config.TFIDF_NGRAM_MAX if ngram_max is None else ngram_max),
        min_df=config.TFIDF_MIN_DF if min_df is None else min_df,
    )


def collapse_to_3(y) -> pd.Series:
    """Colapsa os 5 níveis na escala reduzida de 3 (fácil/médio/difícil).

    Usado na análise complementar de granularidade: junta muito_facil+facil e
    dificil+muito_dificil, mantendo medio. Reduz o desbalanceamento das pontas.
    """
    return pd.Series(list(y)).map(config.COLLAPSE_5_TO_3)


def make_cv(y, n_splits: int | None = None):
    """Cria um divisor de validação cruzada robusto a classes raras.

    Usa ``StratifiedKFold`` quando possível. O nº de folds é limitado pelo
    tamanho da menor classe (não dá para estratificar em k folds se uma classe
    tem menos de k exemplos). Se alguma classe tiver só 1 exemplo, a
    estratificação é impossível e caímos para ``KFold`` simples (com aviso),
    espelhando a lógica de ``split_train_test``.
    """
    contagem = pd.Series(list(y)).value_counts()
    menor = int(contagem.min()) if len(contagem) else 0
    desejado = config.CV_FOLDS if n_splits is None else n_splits

    if menor < 2:
        raros = contagem[contagem < 2].index.tolist()
        k = max(2, min(desejado, max(2, len(contagem))))
        print(
            f"[aviso] validação cruzada SEM estratificação: classes com <2 "
            f"exemplos ({raros}). Usando KFold com k={k}."
        )
        return KFold(n_splits=k, shuffle=True, random_state=config.RANDOM_SEED)

    k = max(2, min(desejado, menor))
    if k < desejado:
        print(f"[aviso] nº de folds reduzido para {k} (menor classe = {menor}).")
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=config.RANDOM_SEED)


def split_train_test(df: pd.DataFrame):
    """Divide treino/teste de forma estratificada e reprodutível.

    Considera apenas linhas rotuladas (questões sem avaliação de aluno ficam de
    fora das etapas supervisionadas). Usado por todas as etapas para que os
    conjuntos sejam idênticos.
    """
    if config.LABEL_COL in df.columns:
        df = df.dropna(subset=[config.LABEL_COL])
        estratificar = df[config.LABEL_COL]
        # A estratificação exige >=2 exemplos por classe. Se algum nível for raro
        # (ex.: 'muito_dificil' com 1 questão), divide sem estratificar.
        contagem = estratificar.value_counts()
        if (contagem < 2).any():
            raros = contagem[contagem < 2].index.tolist()
            print(
                f"[aviso] divisão sem estratificação: classes com <2 exemplos "
                f"({raros}). Considere reunir/ajustar esses níveis."
            )
            estratificar = None
    else:
        estratificar = None
    return train_test_split(
        df,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_SEED,
        stratify=estratificar,
    )


# ----------------------------------------------------------------------------
# Limpeza de texto
# ----------------------------------------------------------------------------

_RE_CODE = re.compile(r"`{1,3}.*?`{1,3}", re.DOTALL)   # trechos `codigo`
_RE_NONWORD = re.compile(r"[^a-z0-9\s]")               # após remover acentos
_RE_SPACE = re.compile(r"\s+")


def clean_text(text: str, remove_stopwords: bool = True) -> str:
    """Normaliza um enunciado: minúsculas, sem acento e sem pontuação.

    Etapas: minúsculas -> remove trechos de código -> remove acentos ->
    remove pontuação -> colapsa espaços -> (opcional) remove stopwords.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = _RE_CODE.sub(" ", text)
    text = _strip_accents(text)
    text = _RE_NONWORD.sub(" ", text)
    text = _RE_SPACE.sub(" ", text).strip()
    if remove_stopwords:
        sw = get_stopwords()
        text = " ".join(t for t in text.split() if t not in sw and len(t) > 1)
    return text


def _strip_accents(text: str) -> str:
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


@lru_cache(maxsize=1)
def get_stopwords() -> frozenset[str]:
    """Stopwords em português + inglês (via NLTK, com fallback embutido)."""
    try:
        import nltk
        from nltk.corpus import stopwords

        try:
            palavras = stopwords.words("portuguese") + stopwords.words("english")
        except LookupError:
            nltk.download("stopwords", quiet=True)
            palavras = stopwords.words("portuguese") + stopwords.words("english")
        # Remove acentos das stopwords p/ casar com o texto já normalizado.
        return frozenset(_strip_accents(p) for p in palavras)
    except Exception:
        return _FALLBACK_STOPWORDS


_FALLBACK_STOPWORDS = frozenset(
    """a o e de da do das dos um uma uns umas para por com que em no na nos nas
    se os as ao aos lhe lhes ele ela eles elas eu tu nos vos como mais mas ou
    ser ter ja nao sim entre sobre seu sua seus suas este esta isto esse essa
    the a an and or of to in is are was were for on with this that be by as at
    it from be been being he she they we you your our""".split()
)
