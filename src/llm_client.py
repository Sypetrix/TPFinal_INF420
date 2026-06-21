"""Cliente fino para a API da Groq (modelos Llama via Groq Cloud).

Centraliza a criação do cliente, o controle de novas tentativas (útil para os
limites da camada gratuita — por minuto: RPM e TPM) e o parsing de respostas em
JSON. É reaproveitado pelos módulos llm_baseline, llm_features, llm_explain,
evaluate e predict.

Por que Groq/Llama? O gargalo da camada gratuita do projeto sempre foi a COTA
(nº de requisições), não dinheiro nem qualidade — a tarefa (classificar
dificuldade e extrair conceitos de enunciados) é leve para um LLM. O modelo
``llama-3.1-8b-instant`` na Groq tem a maior cota diária gratuita disponível
(~14.400 requisições/dia), além de ser muito rápido.

A API da Groq é compatível com o padrão "chat completions" (mensagens com papéis
system/user). Requer o pacote `groq` (veja requirements.txt).
"""
from __future__ import annotations

import json
import time
from typing import Any

from . import config

# Importação preguiçosa: só quem usa LLM precisa do pacote instalado. As etapas
# só de ML (1, 2, 3) rodam sem ter o pacote `groq`.
try:
    from groq import Groq
except ImportError as _err:  # pragma: no cover
    Groq = None
    _IMPORT_ERROR = _err
else:
    _IMPORT_ERROR = None

_client: "Groq | None" = None


def get_client() -> "Groq":
    """Cria (uma única vez) e retorna o cliente da Groq."""
    if Groq is None:
        raise ImportError(
            "Pacote 'groq' não encontrado. Instale com: pip install groq"
        ) from _IMPORT_ERROR
    global _client
    if _client is None:
        _client = Groq(api_key=config.require_api_key())
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
    """True se o erro é estouro da cota DIÁRIA (RPD) — repetir hoje não ajuda.

    Distinto dos limites por minuto (RPM/TPM), esses sim recuperáveis com espera.
    """
    texto = str(err).lower()
    eh_429 = (
        _codigo_http(err) == 429
        or "rate_limit" in texto
        or "429" in texto
    )
    return eh_429 and any(
        s in texto for s in ("per day", "perday", "daily", "requests per day", "rpd")
    )


def _nao_retentavel(err: Exception) -> bool:
    """True para erros de configuração (chave/modelo/permissão): repetir é inútil."""
    if _codigo_http(err) in (400, 401, 403, 404):
        return True
    texto = str(err).lower()
    return any(s in texto for s in (
        "invalid api key", "invalid_api_key", "authentication",
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
    """Envia um prompt ao modelo (Groq) e retorna o texto da resposta.

    Repete com backoff exponencial em erros transitórios (ex.: limites por
    MINUTO — RPM/TPM —, instabilidade). Já em erros de **cota diária (RPD)** ou de
    **configuração** (chave/modelo inválidos), falha imediatamente com mensagem
    clara — repetir nesses casos só gastaria mais cota à toa.

    Obs.: ``as_json`` é mantido para compatibilidade de interface, mas NÃO força o
    modo JSON estrito da Groq (``response_format``), porque alguns prompts do
    projeto pedem um array no topo (lotes), incompatível com esse modo. A
    robustez do JSON vem da temperatura 0 + do parser tolerante ``_parse_json``.
    """
    client = get_client()
    messages: list[dict[str, str]] = []
    if system_instruction:
        messages.append({"role": "system", "content": system_instruction})
    messages.append({"role": "user", "content": prompt})

    kwargs: dict[str, Any] = {
        "model": config.GROQ_MODEL,
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
                    "Limite DIÁRIO de requisições da API da Groq atingido "
                    f"(modelo '{config.GROQ_MODEL}'). Opções: (1) troque "
                    "GROQ_MODEL no .env por outro modelo; (2) reduza a amostra "
                    "com --n; (3) use lotes com --lote; ou (4) tente de novo "
                    "amanhã (a cota renova à meia-noite UTC). Veja os limites em "
                    "https://console.groq.com/docs/rate-limits . "
                    f"Detalhe da API: {err}"
                ) from err
            if _nao_retentavel(err):
                raise RuntimeError(
                    "Erro de configuração da API da Groq (chave inválida, "
                    "modelo inexistente ou sem permissão). Confira GROQ_API_KEY "
                    f"e GROQ_MODEL no .env. Detalhe da API: {err}"
                ) from err
            if attempt == max_retries:
                break
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(
        f"Falha ao chamar a Groq após {max_retries} tentativas "
        f"(modelo '{config.GROQ_MODEL}'): {last_err}"
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
