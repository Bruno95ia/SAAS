# 🧠 SAAS – Sistema de Detecção de Quedas com IA

Este projeto usa **YOLOv8 + CVAT** para detectar quedas de idosos em vídeos.  
Fluxo completo: **Anotação → Conversão → Treinamento → Teste → Pós-processamento → Ajuste**.

---

## 🚀 Guia Rápido

### 1. Anotação no CVAT
```bash
cd ~/cvat
docker compose up -d   # ligar CVAT
#Acesse: http://localhost:8080
	•	Crie Task → importe vídeos.
	•	Labels: Pessoa1, Queda.
	•	Anote com Track (bounding box).
	•	Exporte em YOLO 1.1.
    2. Organizar Dataset

Mover export do CVAT:
    mv ~/Downloads/queda_caso1.zip ~/SAAS/datasets/

Converter para YOLOv8:
    cd ~/SAAS
    source .venv/bin/activate
    python scripts/cvat_to_yolo.py --zip ./datasets/queda_caso1.zip --out ./datasets/queda_caso1_yolo

Estrutura esperada:
datasets/queda_caso1_yolo/
 ├── images/
 ├── labels/
 └── data.yaml

 Treinamento YOLOv8
    yolo detect train data=./datasets/queda_caso1_yolo/data.yaml model=yolov8n.pt epochs=50 imgsz=640
Modelo salvo em:
    runs/detect/train/weights/best.pt
4. Teste rápido
    yolo detect predict model=runs/detect/train/weights/best.pt source=./data/seu_video.mp4 conf=0.5

5. Pós-processamento (filtro temporal)
python scripts/postprocess_falls.py
Saídas:
	•	Vídeo anotado → runs/detect/queda_filtrado.mp4
	•	Log → runs/detect/falls.log
	•	Terminal → [ALERTA] Queda confirmada às ...


6. Ciclo de melhoria
	•	Anotar mais vídeos no CVAT.
	•	Equilibrar quedas e não-quedas.
	•	Repetir passos 2 → 5.
	•	Ajustar:
	•	epochs (mais épocas → melhor aprendizado).
	•	conf (0.4–0.6 → menos falsos positivos).
	•	MIN_CONSECUTIVE_FRAMES (resposta rápida vs conservadora).

Fluxo de evolução

Anotar → Converter → Treinar → Testar → Afinar → Repetir