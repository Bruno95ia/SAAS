# IASENIOR Treinamento via Ultralytics HUB

Este repositório contém os componentes necessários para orquestrar o ciclo de
vida de modelos do projeto **IASENIOR** usando o Ultralytics HUB. A solução
inclui:

* API em FastAPI (`src/saas/train_api.py`) responsável por iniciar treinos,
  acompanhar métricas e disponibilizar o último modelo sincronizado.
* Painel Streamlit (`src/saas/panel_treinamento.py`) para visualização em tempo
  real do backend e acionamento manual dos treinos.
* Script CLI (`src/train_iasenior.py`) usado em pipelines ou execuções locais
  para disparar treinos e manter os logs atualizados.
* Rotina de sincronização (`src/saas/sync_hub_models.py`) que a cada 6 horas
  baixa automaticamente o `best.pt` do HUB.

Todos os módulos compartilham a mesma estrutura de diretórios definida em
`src/saas/utils/iasenior_paths.py`. Em produção o caminho base esperado é
`/mnt/data/SAAS`, mas em desenvolvimento o código detecta o diretório do
repositório automaticamente.

## Requisitos

* Python 3.12
* Dependências listadas em `requirements.txt`
* Token do Ultralytics HUB com permissão para treinar e baixar modelos

### Instalação de dependências

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Configurando o token do HUB

Defina a variável de ambiente `ULTRALYTICS_API_KEY` com o token gerado no
[Ultralytics HUB](https://hub.ultralytics.com/settings?tab=api+keys):

```bash
export ULTRALYTICS_API_KEY="seu_token_aqui"
```

Opcionalmente ajuste os endpoints se estiver utilizando instâncias privadas:

```bash
export ULTRALYTICS_HUB_API="https://seu-endpoint-api"
export ULTRALYTICS_HUB_WEB="https://seu-endpoint-web"
```

## Estrutura de diretórios

* `logs/treinos.csv`: histórico de execuções, atualizados pela API, CLI e
  sincronizador.
* `logs/train_progress.json`: estado atual do treino em execução.
* `logs/model_sync.json`: metadados do último `best.pt` baixado.
* `models/`: contém o modelo ativo (`iasenior_best.pt`) e backups em `models/archive/`.
* `datasets/`: repositório local para datasets exportados no formato YOLO.

Os diretórios são criados automaticamente na primeira execução de qualquer
componente.

## Executando a API FastAPI

```bash
uvicorn saas.train_api:app --host 0.0.0.0 --port 8000 --reload
```

Endpoints principais:

* `POST /api/train/start` – inicia treino remoto no HUB.
* `GET /api/train/status` – retorna andamento, métricas e URLs de pesos.
* `GET /api/models/latest` – informa o arquivo do modelo sincronizado.

## Painel Streamlit

Após subir a API, inicie o painel com:

```bash
streamlit run src/saas/panel_treinamento.py
```

Recursos do painel:

* Indicador de status do backend.
* Formulário para iniciar novos treinos (dataset, épocas, modelo, observações).
* Visualização de métricas (mAP, perda, precisão, recall) e barra de progresso.
* Botão para baixar o último modelo sincronizado.
* Histórico completo das execuções.

## Script CLI

Use o utilitário para iniciar um treino diretamente pelo terminal ou pipeline:

```bash
python src/train_iasenior.py <dataset_id> --epochs 100 --model yolov8m.pt
```

Argumentos relevantes:

* `--api-key`: sobrescreve a variável de ambiente.
* `--fallback-local`: caso o HUB esteja indisponível, realiza treino local usando
  o dataset `datasets/<dataset_id>/data.yaml`.
* `--notes`: adiciona anotações no CSV de histórico.

## Sincronização periódica

Para manter o backend sempre com o modelo mais recente do HUB, execute:

```bash
python src/saas/sync_hub_models.py
```

O script executa uma sincronização imediata e agenda novas verificações a cada
6 horas (utiliza a biblioteca `schedule`). O arquivo baixado é salvo como
`models/iasenior_best.pt` e versões anteriores são movidas para `models/archive/`.

## Logs e monitoramento

* `train_progress.json` é atualizado sempre que uma requisição de status é
  realizada – tanto pela API quanto pelo painel.
* `treinos.csv` consolida eventos de início de treino, finalização local e
  sincronizações de modelo.
* `model_sync.json` contém metadados da última sincronização, útil para depurar
  jobs externos.

## Dicas adicionais

* Ajuste o intervalo do painel com a variável `IASENIOR_PANEL_REFRESH` (segundos).
* Redirecione a API consumida pelo painel definindo `IASENIOR_TRAIN_API`.
* Para upload de datasets via CLI, utilize diretamente a função
  `UltralyticsHubManager.upload_dataset` no REPL Python ou scripts auxiliares.

Com esses componentes integrados você terá um fluxo completo de treino e
monitoramento do IASENIOR totalmente conectado ao Ultralytics HUB.
