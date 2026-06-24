"""Cliente fino e MODULAR para LLMs com API no padrão OpenAI.

Suporta múltiplos provedores que expõem o endpoint de *chat completions*
compatível com OpenAI — basta trocar base_url/chave/modelo. O provedor ativo é
escolhido por ``LLM_PROVIDER`` no ``.env``:

  * ``groq``     (padrão) — Llama via Groq Cloud; maior cota gratuita diária.
  * ``deepseek`` — modelos DeepSeek (alternativa, também gratuita/barata).

Trocar de provedor é uma linha no ``.env`` (``LLM_PROVIDER`` e a chave
correspondente), sem mexer no código. Centraliza também o controle de novas
tentativas (limites por minuto da camada gratuita) e o parsing tolerante de JSON.

Requer o pacote ``openai`` (veja requirements.txt). As etapas só de ML rodam sem
ele instalado (import preguiçoso).
"""
from __future__ import annotations

import json
import time
from typing import Any

from . import config

# Importação preguiçosa: só quem usa LLM precisa do pacote instalado.
try:
    from openai import OpenAI
except ImportError as _err:  # pragma: no cover
    OpenAI = None
    _IMPORT_ERROR = _err
else:
    _IMPORT_ERROR = None

_client: "OpenAI | None" = None


def get_client() -> "OpenAI":
    """Cria (uma única vez) e retorna o cliente do provedor ativo."""
    if OpenAI is None:
        raise ImportError(
            "Pacote 'openai' não encontrado. Instale com: pip install openai"
        ) from _IMPORT_ERROR
    global _client
    if _client is None:
        _client = OpenAI(api_key=config.require_api_key(), base_url=config.LLM_BASE_URL)
    return _client


def _codigo_http(err: Exception) -> int | None:
    """Tenta extrair o código HTTP do erro do SDK (429, 404, ...)."""
    for attr in ("status_code", "code"):
        v = getattr(err, attr, None)
        if isinstance(v, int):
            return v
    resp = getattr(err, "response", None)
    v = getattr(resp, "status_code", None)
    return v if isinstance(v, int) else None


def _eh_cota_diaria(err: Exception) -> bool:
    """True se o erro é estouro da cota DIÁRIA (RPD) — repetir hoje não ajuda."""
    texto = str(err).lower()
    eh_429 = _codigo_http(err) == 429 or "rate_limit" in texto or "429" in texto
    return eh_429 and any(
        s in texto for s in ("per day", "perday", "daily", "requests per day", "rpd")
    )


def _nao_retentavel(err: Exception) -> bool:
    """True para erros de configuração (chave/modelo/permissão): repetir é inútil."""
    if _codigo_http(err) in (400, 401, 403, 404):
        return True
    texto = str(err).lower()
    return any(s in texto for s in (
        "invalid api key", "invalid_api_key", "authentication", "unauthorized",
        "permission", "not found", "does not exist", "model_not_found",
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
    """Envia um prompt ao LLM do provedor ativo e retorna o texto da resposta.

    Repete com backoff exponencial em erros transitórios (limites por MINUTO,
    instabilidade). Em cota DIÁRIA (RPD) ou erro de configuração (chave/modelo),
    falha imediatamente com mensagem clara.

    Obs.: ``as_json`` é mantido para compatibilidade, mas não força o modo JSON
    estrito (alguns prompts pedem array no topo, incompatível com esse modo). A
    robustez vem da temperatura 0 + parser tolerante ``_parse_json``.
    """
    client = get_client()
    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": config.LLM_MODEL,
        "messages": messages,
        "temperature": temperature,
    }
    if max_output_tokens:
        kwargs["max_tokens"] = max_output_tokens

    delay = 2.0
    last_err: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()
        except Exception as err:  # a API pode lançar vários tipos de exceção
            last_err = err
            if _eh_cota_diaria(err):
                raise RuntimeError(
                    "Limite DIÁRIO de requisições da API atingido "
                    f"(provedor '{config.LLM_PROVIDER}', modelo '{config.LLM_MODEL}'). "
                    "Opções: (1) troque LLM_MODEL/LLM_PROVIDER no .env; (2) reduza a "
                    "amostra com --n; (3) use --lote; ou (4) tente amanhã. "
                    f"Detalhe da API: {err}"
                ) from err
            if _nao_retentavel(err):
                raise RuntimeError(
                    "Erro de configuração da API do LLM (chave inválida, modelo "
                    "inexistente ou sem permissão). Confira LLM_PROVIDER, a chave do "
                    f"provedor e LLM_MODEL no .env. Detalhe da API: {err}"
                ) from err
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(
        f"Falha ao chamar o LLM após {max_retries} tentativas "
        f"(provedor '{config.LLM_PROVIDER}', modelo '{config.LLM_MODEL}'): {last_err}"
    )


def generate_json(
    prompt: str,
    system_instruction: str | None = None,
    **kwargs: Any,
) -> Any:
    """Igual a generate(), mas já faz o parsing do JSON retornado.

    Se o LLM devolver algo que NÃO é JSON válido (mesmo após o parser tolerante
    de ``_parse_json``), retorna ``None`` em vez de levantar exceção. Isso
    permite que os chamadores (extract_batch / classify_batch) acionem o
    fallback item-a-item — o prompt individual é bem menor e tem chance muito
    maior de produzir JSON válido. Ou seja: 1 lote ruim não trava o pipeline.
    """
    raw = generate(prompt, system_instruction, as_json=True, **kwargs)
    try:
        return _parse_json(raw)
    except json.JSONDecodeError as err:
        # Log curto para diagnóstico (não inunda o terminal).
        print(f"[aviso] LLM devolveu JSON inválido ({err}); "
              "caindo p/ fallback item-a-item neste lote.")
        return None


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
        for open_c, close_c in (("{", "}"), ("[", "]")):
            start, end = text.find(open_c), text.rfind(close_c)
            if start != -1 and end > start:
                try:
                    return json.loads(text[start : end + 1])
                except json.JSONDecodeError:
                    continue
        raise
