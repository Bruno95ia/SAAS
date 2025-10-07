# 📄 Relatório de Funcionalidades do Sistema de Detecção de Quedas

## ✅ Funcionalidades já desenvolvidas

### 🔹 1. Anotação de dados
- Uso do **CVAT** para criar datasets anotados com labels `Pessoa1` e `Queda`.
- Exportação para formato YOLO (`datasets/queda_casoX_yolo/` com `images/` e `labels/` divididos em `train/val/test`).
- Execução de múltiplas tarefas (ex.: `Queda_caso1`, `Queda_caso2`, etc.).

---

### 🔹 2. Treinamento do modelo
- Treinamento de um **YOLOv8** para detecção de quedas.
- Estrutura automática (`runs/detect/train/...`) com métricas e gráficos.
- Configuração de hiperparâmetros (epochs, img size, batch size, etc.).
- Possibilidade de **re-treinar e afinar** com novos datasets.
- Suporte para **validação automática** (`val/`) durante treino.

---

### 🔹 3. Inferência / Testes
- Predição em:
  - **Imagens** (`val_batch0_pred.jpg`, etc.).
  - **Vídeos** (`runs/detect/predict/*.mp4`).
- Testes diretos com vídeos reais de quedas (ex.: `Queda_banheiro.mp4`, `Queda_cadeira.mp4`).

---

### 🔹 4. Pós-processamento de quedas
- Script `postprocess_falls.py` que:
  - Analisa os resultados frame a frame.
  - Confirma **quedas persistentes** com parâmetros configuráveis:
    - `min_consecutive_frames`
    - `min_clear_frames`
    - `alert_cooldown_s`
  - Gera log detalhado (`runs/detect/falls.log`).
  - Produz vídeos filtrados com quedas confirmadas (`*_filtrado.mp4`).

---

### 🔹 5. Configuração flexível via `config.yaml`
- Centralização dos parâmetros do sistema:
  - ⚙️ **Runtime**: porta de métricas, device (CPU/GPU).
  - 🎥 **Vídeos**: diretório inteiro (`./data/`) ou arquivo único.
  - 🔐 **Privacy**: salvar ou não clipes, definir janelas antes/depois da queda.
  - 🧍 **Pose**: modelo de pose (`yolov8n-pose.pt`) integrado.
  - 🔔 **Logic**: regras de confirmação de quedas.
  - 📢 **Alerts**: logs (estrutura pronta para webhook/email).

---

### 🔹 6. Aplicativo principal (`app.py`)
- Processa todos os vídeos do diretório configurado.
- Para cada vídeo:
  - Detecta pessoas.
  - Extrai keypoints (pose).
  - Aplica lógica de confirmação de quedas.
  - Dispara eventos JSON no terminal, ex.:
    ```json
    {"event":"fall_alert","prob":0.91,"snapshot":"runs/detect/snapshots/Queda_banheiro/20250929_224501_frame000342.jpg"}
    ```
- Exporta vídeos anotados (`*_annot.mp4`) em `runs/detect/annotated/`.

---

### 🔹 7. Snapshots automáticos
- A cada alerta confirmado:
  - Salva **frame JPG** (`runs/detect/snapshots/...`).
  - Inclui caminho no JSON/log.
- Snapshots organizados por pasta do vídeo + timestamp.

---

## 🚧 Funcionalidades em desenvolvimento / próximos passos
1. 🔴 **Reduzir falsos positivos** → tunar parâmetros (`prob_threshold`, `min_frames`, etc.) e enriquecer dataset.
2. 🎞 **Gerar GIFs curtos** (pré/pós-queda) junto com snapshots.
3. 🌐 **Sistema de alertas em tempo real**: webhook, email, ou dashboard.
4. 📊 **Dashboard de métricas** (quedas por hora, precisão, recall, etc.).
5. 🖥️ **Pipeline em tempo real** (câmera IP ou webcam).

---

👉 Situação atual: já existe um **pipeline funcional de detecção de quedas** (dados → treino → inferência → alerta → snapshot).  
Agora a prioridade pode ser **reduzir falsos positivos** ou **migrar para tempo real**.
