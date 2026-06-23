<#
.SYNOPSIS
  Roda o pipeline completo do TP (uma ou mais bases) e SALVA todos os
  resultados em .txt na pasta resultados\, para conferência e montagem do banner.

.DESCRIPTION
  Para cada base (ex.: Neps, INF110) executa, em ordem:
    1. ingest        (consolida arquivos/<base> -> data/raw/<base>/questoes.csv)
    2. preprocess    (limpeza + TF-IDF)               [offline]
    3. train_ml      (5 níveis)                        [offline]
    4. train_ml      (3 níveis)                        [offline]
    5. figuras       (PNGs do banner)                  [offline]
    6. llm_concepts  (conceitos via LLM -> feature C)  [API]  (pulado com -SemLLM)
    7. llm_baseline  (classificação direta via LLM)    [API]  (pulado com -SemLLM)
    8. evaluate      (A: ML | B: LLM | C: ML+conceitos)[API/offline]
  No fim, consolida tudo em resultados\resumo_banner.txt.

  Cada etapa grava a saída em resultados\<etapa>_<base>.txt E mostra no console.
  As etapas com [API] consomem cota da Groq; as [offline] não usam internet.

.PARAMETER Bases
  Bases a processar. Padrão: Neps, INF110.
.PARAMETER Lote
  Enunciados por requisição no LLM (prompt packing). Padrão 10 (~10x menos chamadas).
.PARAMETER Sleep
  Pausa (s) entre chamadas à API, evita estourar o limite por minuto. Padrão 3.
.PARAMETER AmostraN
  Tamanho da amostra de teste no evaluate/llm_baseline. Padrão 40 (uso moderado).
.PARAMETER ConceitosN
  Quantos enunciados extrair conceitos. 0 = base toda (recomendado p/ approach C).
.PARAMETER SemLLM
  Roda só as etapas offline (sem gastar cota da API). evaluate roda com --no-llm.
.PARAMETER ComAvaliar
  Também roda a inferência (predict_difficulty) sobre a fonte 'avaliar'.

.EXAMPLE
  .\rodar_tudo.ps1                         # tudo, as duas bases, uso moderado
  .\rodar_tudo.ps1 -SemLLM                 # só offline (sem API)
  .\rodar_tudo.ps1 -Bases Neps -AmostraN 60
#>
[CmdletBinding()]
param(
    [string[]]$Bases      = @('Neps','INF110'),
    [int]     $Lote       = 10,
    [double]  $Sleep      = 3,
    [int]     $AmostraN   = 40,
    [int]     $ConceitosN = 0,
    [switch]  $SemLLM,
    [switch]  $ComAvaliar
)

$ErrorActionPreference = 'Continue'
$ProjRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjRoot

# --- UTF-8: evita texto embaralhado (acentos) no console e nos .txt ---
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUTF8       = '1'
try { [Console]::OutputEncoding = [System.Text.Encoding]::UTF8 } catch {}
$OutputEncoding = [System.Text.Encoding]::UTF8

# --- pasta de resultados ---
$ResDir = Join-Path $ProjRoot 'resultados'
New-Item -ItemType Directory -Force -Path $ResDir | Out-Null

# --- ativa o ambiente virtual, se existir (.venv ou venv) ---
foreach ($v in @('.venv','venv')) {
    $act = Join-Path $ProjRoot "$v\Scripts\Activate.ps1"
    if (Test-Path $act) { Write-Host "Ativando ambiente: $v" -ForegroundColor DarkGray; . $act; break }
}

# --- escolhe o executável Python ---
$Py = if (Get-Command python -ErrorAction SilentlyContinue) { 'python' } else { 'py' }

function Run-Step {
    param([string]$Titulo, [string]$Etapa, [string]$Base, [string[]]$ArgsList)
    $log = Join-Path $ResDir ("{0}_{1}.txt" -f $Etapa, $Base)
    Write-Host ""
    Write-Host ("=== [{0}] {1} ===" -f $Base, $Titulo) -ForegroundColor Cyan
    Write-Host ("  > $Py -m $($ArgsList -join ' ')  (log: resultados\$(Split-Path $log -Leaf))") -ForegroundColor DarkGray
    # Cabeçalho + saída do comando passam por UM único Tee-Object (sem -Append,
    # que não existe no Windows PowerShell 5.1), gravando no .txt e no console.
    # O ForEach { "$_" } converte cada linha (inclusive o stderr/barra de progresso
    # do tqdm) em texto puro — assim o PowerShell 5.1 não pinta o stderr de vermelho
    # como "NativeCommandError" (que NÃO é um erro de verdade, é só a barra de progresso).
    & {
        "### $Titulo | base=$Base | $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
        "### comando: $Py -m $($ArgsList -join ' ')"
        ""
        & $Py -m @ArgsList 2>&1 | ForEach-Object { "$_" }
    } | Tee-Object -FilePath $log
    if ($LASTEXITCODE -ne 0) {
        Write-Host ("  [ERRO] etapa '{0}' (base {1}) retornou código {2}. Veja o log." -f $Etapa,$Base,$LASTEXITCODE) -ForegroundColor Red
    }
}

Write-Host "############################################################" -ForegroundColor Yellow
Write-Host "# Pipeline TP INF420 — bases: $($Bases -join ', ')"            -ForegroundColor Yellow
Write-Host "# Lote=$Lote  Sleep=$Sleep  AmostraN=$AmostraN  SemLLM=$SemLLM" -ForegroundColor Yellow
Write-Host "############################################################" -ForegroundColor Yellow

foreach ($Base in $Bases) {
    $env:DATASET = $Base
    Write-Host "`n>>> BASE ATIVA: $Base (DATASET=$Base)" -ForegroundColor Green

    # ---- Offline (sem API) ----
    Run-Step "Ingestão"               "1_ingest"        $Base @('src.ingest')
    Run-Step "Pré-processamento"      "2_preprocess"    $Base @('src.preprocess')
    Run-Step "Treino ML (5 níveis)"   "3_train_5niveis" $Base @('src.train_ml')
    Run-Step "Treino ML (3 níveis)"   "4_train_3niveis" $Base @('src.train_ml','--niveis','3')
    Run-Step "Figuras (PNG do banner)" "5_figuras"      $Base @('src.figuras')

    if (-not $SemLLM) {
        # ---- Etapas com LLM (consomem cota da Groq) ----
        $concArgs = @('src.llm_concepts','--lote',"$Lote",'--sleep',"$Sleep")
        if ($ConceitosN -gt 0) { $concArgs += @('--n',"$ConceitosN") }
        Run-Step "Conceitos via LLM"   "6_llm_concepts"  $Base $concArgs

        Run-Step "Baseline LLM (few-shot)" "7_llm_baseline" $Base `
            @('src.llm_baseline','--n',"$AmostraN",'--few-shot','--lote',"$Lote",'--sleep',"$Sleep")

        Run-Step "Avaliação A/B/C"     "8_evaluate"      $Base `
            @('src.evaluate','--n',"$AmostraN",'--lote',"$Lote",'--sleep',"$Sleep")
    }
    else {
        Run-Step "Avaliação A/C (offline)" "8_evaluate_offline" $Base `
            @('src.evaluate','--n',"$AmostraN",'--no-llm')
    }
}

# ---- Inferência opcional sobre a fonte 'avaliar' ----
if ($ComAvaliar) {
    $env:DATASET = $Bases[0]   # usa o classificador da 1a base
    & $Py -m src.ingest *> $null; $env:DATASET = 'avaliar'; & $Py -m src.ingest *> $null
    $env:DATASET = $Bases[0]
    $predArgs = @('src.predict_difficulty','--fonte','avaliar')
    if ($SemLLM) { $predArgs += '--no-llm' } else { $predArgs += @('--lote',"$Lote") }
    Run-Step "Inferência (avaliar)" "9_predict_avaliar" $Bases[0] $predArgs
}

# ---- Consolida tudo num .txt para o banner ----
Write-Host "`n=== Consolidando resultados -> resultados\resumo_banner.txt ===" -ForegroundColor Cyan
& $Py -m src.coletar_resultados 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath (Join-Path $ResDir 'resumo_banner_log.txt')

Write-Host "`nPRONTO. Confira a pasta 'resultados\' (resumo_banner.txt) e 'figuras\'." -ForegroundColor Green
