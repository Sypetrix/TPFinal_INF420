"""Cliente fino para o Google Gemini (Google AI Studio).

Centraliza a criação do cliente, o controle de novas tentativas (útil para o
limite de requisições por minuto da camada gratuita) e o parsing de respostas
em JSON. É reaproveitado pelos módulos llm_baseline, llm_features e llm_explain.

Requer o pacote `google-genai` (veja requirements.txt).
"""
from __future__ import annotations

import json
import time
from typing import Any

from . import config

# Importação preguiçosa: só quem usa LLM precisa do pacote instalado.
try:
    from google import genai
    from google.genai import types
except ImportError as _err:  # pragma: no cover
    genai = None
    types = None
    _IMPORT_ERROR = _err
else:
    _IMPORT_ERROR = None

_client: "genai.Client | None" = None


def get_client() -> "genai.Client":
    """Cria (uma única vez) e retorna o cliente do Gemini."""
    if genai is None:
        raise ImportError(
            "Pacote 'google-genai' não encontrado. Instale com: "
            "pip install google-genai"
        ) from _IMPORT_ERROR
    global _client
    if _client is None:
        _client = genai.Client(api_key=config.require_api_key())
    return _client


def generate(
    prompt: str,
    system_instruction: str | None = None,
    *,
    temperature: float = 0.0,
    as_json: bool = False,
    max_output_tokens: int | None = None,
    max_retries: int = 5,
) -> str:
    """Envia um prompt ao Gemini e retorna o texto da resposta.

    Faz novas tentativas com backoff exponencial em caso de erro de API
    (ex.: estouro do limite de requisições por minuto).
    """
    client = get_client()
    cfg = types.GenerateContentConfig(
        temperature=temperature,
        system_instruction=system_instruction,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json" if as_json else None,
    )

    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.models.generate_content(
                model=config.GEMINI_MODEL,
                contents=prompt,
                config=cfg,
            )
            return (resp.text or "").strip()
        except Exception as err:  # a API pode lançar vários tipos de exceção
            last_err = err
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(
        f"Falha ao chamar o Gemini após {max_retries} tentativas: {last_err}"
    )


def generate_json(
    prompt: str,
    system_instruction: str | None = None,
    **kwargs: Any,
) -> Any:
    """Igual a generate(), mas já faz o parsing do JSON retornado."""
    raw = generate(prompt, system_instruction, as_json=True, **kwargs)
    return _parse_json(raw)


def _parse_json(raw: str) -> Any:
    """Carrega JSON tolerando cercas de código (```json ... ```)."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Última tentativa: recorta do primeiro '{'/'[' ao último '}'/']'.
        for open_c, close_c in (("{", "}"), ("[", "]")):
            start, end = text.find(open_c), text.rfind(close_c)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise
