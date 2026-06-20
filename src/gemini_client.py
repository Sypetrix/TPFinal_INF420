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


def _codigo_http(err: Exception) -> int | None:
    """Tenta extrair o código HTTP do erro do SDK (429, 404, ...)."""
    for attr in ("code", "status_code"):
        v = getattr(err, attr, None)
        if isinstance(v, int):
            return v
    return None


def _eh_cota_diaria(err: Exception) -> bool:
    """True se o erro é estouro da cota DIÁRIA (RPD) — repetir hoje não ajuda.

    Distinto do limite por minuto (RPM), esse sim recuperável com espera.
    """
    texto = str(err).lower()
    eh_429 = _codigo_http(err) == 429 or "resource_exhausted" in texto or "429" in texto
    return eh_429 and any(s in texto for s in ("per day", "perday", "daily", "requests per day"))


def _nao_retentavel(err: Exception) -> bool:
    """True para erros de configuração (chave/modelo/permissão): repetir é inútil."""
    if _codigo_http(err) in (400, 401, 403, 404):
        return True
    texto = str(err).lower()
    return any(s in texto for s in (
        "api key not valid", "api_key_invalid", "unauthenticated",
        "permission", "not found", "is not supported",
    ))


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

    Repete com backoff exponencial em erros transitórios (ex.: limite por
    MINUTO/RPM, instabilidade). Já em erros de **cota diária (RPD)** ou de
    **configuração** (chave/modelo inválidos), falha imediatamente com mensagem
    clara — repetir nesses casos só gastaria mais cota à toa.
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
            if _eh_cota_diaria(err):
                raise RuntimeError(
                    "Limite DIÁRIO de requisições da API do Gemini atingido "
                    f"(modelo '{config.GEMINI_MODEL}'). Opções: (1) troque "
                    "GEMINI_MODEL no .env por um modelo com cota gratuita maior "
                    "(ex.: gemini-2.0-flash ou gemini-2.5-flash-lite); (2) reduza "
                    "a amostra com --n; (3) habilite faturamento no Google AI "
                    "Studio; ou (4) tente de novo amanhã. Veja os limites em "
                    "https://ai.google.dev/gemini-api/docs/rate-limits . "
                    f"Detalhe da API: {err}"
                ) from err
            if _nao_retentavel(err):
                raise RuntimeError(
                    "Erro de configuração da API do Gemini (chave inválida, "
                    "modelo inexistente ou sem permissão). Confira GEMINI_API_KEY "
                    f"e GEMINI_MODEL no .env. Detalhe da API: {err}"
                ) from err
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(
        f"Falha ao chamar o Gemini após {max_retries} tentativas "
        f"(modelo '{config.GEMINI_MODEL}'): {last_err}"
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
