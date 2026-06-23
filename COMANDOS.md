# Comandos para rodar os testes e coletar dados do banner

Todos os comandos rodam da **raiz do projeto**, com o ambiente virtual ativo.
As saídas vão para `resultados/` (cada etapa num `.txt`) e o resumo final fica em
`resultados/resumo_banner.txt`.

## Opção rápida (recomendada): script único

```powershell
# tudo, as duas bases (Neps + INF110), uso moderado de API, salva em resultados\
.\rodar_tudo.ps1

# só as etapas offline (NÃO gasta cota da Groq)
.\rodar_tudo.ps1 -SemLLM

# uma base só, amostra de teste maior
.\rodar_tudo.ps1 -Bases Neps -AmostraN 60

# também rodar a inferência sobre a fonte 'avaliar'
.\rodar_tudo.ps1 -ComAvaliar
```

> Se o PowerShell bloquear o script:
> `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` e rode de novo.

---

## Opção manual (comando a comando)

### 0. Preparação (uma vez)

```powershell
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
# .env já está configurado (DATASET=Neps; chave da Groq preenchida)
```

### 1. Pipeline OFFLINE (sem API) — gera as métricas de ML do banner

Faça para **cada base** trocando o `DATASET`:

```powershell
$env:DATASET = "Neps"          # ou "INF110"
python -m src.ingest                       # consolida a base
python -m src.preprocess                   # limpeza + TF-IDF
python -m src.train_ml                     # modelos, 5 níveis
python -m src.train_ml --niveis 3          # modelos, 3 níveis
python -m src.figuras                      # PNGs em figuras\
```

Saídas (por base): `models\<base>\ml_metrics.csv`, `ml_metrics_3niveis.csv`,
`matriz_confusao*.csv` e `figuras\*_<base>.png`.

### 2. Etapas com LLM (consomem cota da Groq) — comparação A/B/C do banner

```powershell
$env:DATASET = "Neps"          # ou "INF110"
# conceitos via LLM (feature da abordagem C). Lote 10 = ~10x menos chamadas.
python -m src.llm_concepts --lote 10 --sleep 3
# baseline: LLM classifica direto (few-shot)
python -m src.llm_baseline --n 40 --few-shot --lote 10 --sleep 3
# avaliação final comparando ML puro (A) x LLM (B) x ML+conceitos (C)
python -m src.evaluate --n 40 --lote 10 --sleep 3
```

Saída: `models\<base>\comparacao_final.csv` e `data\processed\<base>\llm_*.csv`.

> **Sem querer gastar cota?** Rode só a parte offline da avaliação:
> `python -m src.evaluate --n 40 --no-llm` (compara A e C, sem chamar a API).

### 3. (Opcional) Inferência sobre questões novas

```powershell
$env:DATASET = "avaliar"; python -m src.ingest      # consolida a fonte 'avaliar'
$env:DATASET = "Neps"                                # usa o classificador do Neps
python -m src.predict_difficulty --fonte avaliar --lote 10     # completo (+ conceitos)
python -m src.predict_difficulty --fonte avaliar --no-llm      # offline
```

### 4. Consolidar tudo num .txt para conferir

```powershell
python -m src.coletar_resultados            # gera resultados\resumo_banner.txt
```

---

## O que mudou no código (otimizações)

- **`.env`**: `DATASET` estava com uma lista inválida (`INF110, NEPS, OBI, SPOJ`);
  corrigido para uma fonte única (`Neps`). Troque por base via `$env:DATASET`.
- **Lotes maiores por padrão**: o `--lote` padrão das etapas com LLM
  (`llm_concepts`, `llm_baseline`, `evaluate`, `predict_difficulty`) passou de
  **1 para 10** — reduz ~10x o nº de chamadas à API para a mesma tarefa. Use
  `--lote 1` se quiser o modo item-a-item.
- **`--sleep` no `evaluate`**: agora dá para espaçar as chamadas (evita estourar o
  limite por minuto da Groq com lotes grandes).
- **Novo `src.coletar_resultados`**: lê os CSVs gerados e escreve um resumo único e
  legível em `resultados/resumo_banner.txt`.

## Onde estão os números do banner

| Seção do banner | Arquivo |
|---|---|
| Tabela de modelos (ML) | `models\<base>\ml_metrics.csv` / `..._3niveis.csv` |
| Comparação A/B/C | `models\<base>\comparacao_final.csv` |
| Matriz de confusão | `models\<base>\matriz_confusao*.csv` |
| Figuras (2 do banner) | `figuras\comparacao_f1_<base>.png`, `figuras\matriz_confusao_<base>.png` |
| **Resumo de tudo** | `resultados\resumo_banner.txt` |
