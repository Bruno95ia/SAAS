import cv2
from ultralytics import YOLO
from datetime import datetime

# Configurações
MODEL_PATH = "runs/detect/train/weights/best.pt"
VIDEO_PATH = "data/Queda_banheiro3.mp4"
OUTPUT_PATH = "runs/detect/queda_filtrado.mp4"
LOG_PATH = "runs/detect/falls.log"

# Parâmetros do filtro temporal
MIN_CONSECUTIVE_FRAMES = 8 # mínimo de frames seguidos para confirmar queda
CONFIDENCE_THRESHOLD = 0.6  # confiança mínima da detecção

# Carregar modelo
model = YOLO(MODEL_PATH)

# Abrir vídeo
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

# Estado de detecção
consecutive_falls = 0
fall_confirmed = False
alerts_sent = 0

# Abrir log
log_file = open(LOG_PATH, "a")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Rodar predição YOLO no frame
    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, verbose=False)

    detected_fall = False
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            label = model.names[cls_id]

            if label.lower() == "queda" and conf >= CONFIDENCE_THRESHOLD:
                detected_fall = True
                # desenhar caixa no frame
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, f"Queda {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Controle de frames consecutivos
    if detected_fall:
        consecutive_falls += 1
    else:
        consecutive_falls = 0

    # Confirmar queda se atingir limite
    if consecutive_falls >= MIN_CONSECUTIVE_FRAMES and not fall_confirmed:
        fall_confirmed = True
        alerts_sent += 1
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        msg = f"[ALERTA] Queda confirmada às {timestamp} (alerta #{alerts_sent})"
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # Mostrar status no vídeo
    if fall_confirmed:
        cv2.putText(frame, "QUEDA CONFIRMADA!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    out.write(frame)

cap.release()
out.release()
log_file.close()
print(f"[INFO] Vídeo processado salvo em: {OUTPUT_PATH}")
print(f"[INFO] Log de alertas salvo em: {LOG_PATH}")
