import cv2
import time
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms, models

# ── Configurações ────────────────────────────────────────────────────────
EMOTIONS = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
MODEL_PATH = "resnet18_fer_finetuned.pth"   # Ajuste o caminho se necessário
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Cores por emoção (BGR)
EMOTION_COLORS = {
    "Angry":    (0,   0,   220),
    "Disgust":  (0,   128, 0  ),
    "Fear":     (128, 0,   128),
    "Happy":    (0,   220, 220),
    "Sad":      (220, 80,  0  ),
    "Surprise": (0,   200, 255),
    "Neutral":  (180, 180, 180),
}

# ── Carrega modelo ResNet-18 fine-tuned ──────────────────────────────────
def load_model(path, device):
    m = models.resnet18(weights=None)
    m.fc = nn.Sequential(
        nn.Linear(512, 256), nn.ReLU(), nn.Dropout(0.4), nn.Linear(256, 7)
    )
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval().to(device)
    return m

# ── Transform para inferência ────────────────────────────────────────────
INF_TRANSFORM = transforms.Compose([
    transforms.Grayscale(num_output_channels=3), # Converte nativamente L para RGB
    transforms.Resize((224, 224)),               # Formato quadrado exigido pela ResNet
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

@torch.no_grad()
def predict_emotion(model, face_bgr, device):
    """Recebe recorte BGR (numpy), retorna emoção e confiança."""
    gray = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2GRAY)
    pil  = Image.fromarray(gray, mode="L")
    inp  = INF_TRANSFORM(pil).unsqueeze(0).to(device)
    out  = model(inp)
    prob = torch.softmax(out, dim=1)[0]
    idx  = prob.argmax().item()
    return EMOTIONS[idx], float(prob[idx]) * 100

def draw_bar(frame, x, y, w, emotion, confidence, color):
    """Desenha barra de confiança e rótulo."""
    label = f"{emotion}  {confidence:.0f}%"
    bar_w = int(w * confidence / 100)
    
    # Fundo semitransparente
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y - 30), (x + w, y), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Barra de percentagem
    cv2.rectangle(frame, (x, y - 8), (x + bar_w, y - 2), color, -1)
    cv2.putText(frame, label, (x + 4, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)


# ── Lógica Principal ─────────────────────────────────────────────────────
def main():
    print("A carregar modelo...")
    model = load_model(MODEL_PATH, DEVICE)
    print(f"Modelo carregado ({DEVICE}).")

    # Detetor nativo do OpenCV (extremamente rápido)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Não foi possível aceder à câmara.")

    # Forçar parâmetros da câmara para evitar limite de FPS do Windows
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    fps_time = time.time()
    frame_skip = 0
    
    # Variáveis para guardar o estado do frame anterior
    last_faces = ()
    last_emotion = "Neutral"
    last_conf = 0.0

    # --- CONFIGURAÇÃO PARA ECRÃ INTEIRO ---
    window_name = "Deteccao de Emocoes em Tempo Real"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    print("Câmara iniciada. Pressione Q para sair.")

    while True:
        ret, frame = cap.read()
        if not ret: 
            break

        frame_skip += 1
        
        # OTIMIZAÇÃO: Deteta e prevê a emoção apenas 1 a cada 3 frames
        if frame_skip % 3 == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # OTIMIZAÇÃO: Reduz a imagem em 50% para procurar o rosto muito mais rápido
            small_gray = cv2.resize(gray_frame, (0, 0), fx=0.5, fy=0.5)
            faces_reduzidas = face_cascade.detectMultiScale(small_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
            
            # Multiplica por 2 para restaurar as coordenadas para a imagem original
            last_faces = [(x*2, y*2, w*2, h*2) for (x, y, w, h) in faces_reduzidas]

            # Pega a emoção do primeiro rosto encontrado
            if len(last_faces) > 0:
                x, y, w, h = last_faces[0]
                fh, fw = frame.shape[:2]
                
                # Garante que as coordenadas não saem fora do limite do ecrã
                x1, y1 = max(0, x), max(0, y)
                x2, y2 = min(fw, x+w), min(fh, y+h)
                
                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size > 0:
                    last_emotion, last_conf = predict_emotion(model, face_crop, DEVICE)

        # Desenha a caixa em TODOS os frames (mesmo os não processados) para fluidez visual
        if len(last_faces) > 0:
            x, y, w, h = last_faces[0]
            color = EMOTION_COLORS.get(last_emotion, (200, 200, 200))
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            draw_bar(frame, x, y+h + 30, w, last_emotion, last_conf, color)

        # Cálculo do FPS Real
        fps = 1.0 / (time.time() - fps_time + 1e-9)
        fps_time = time.time()
        
        cv2.putText(frame, f"FPS: {fps:.0f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Pressione Q para sair", (10, frame.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Mostra a imagem na janela configurada para full screen
        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("A encerrar.")

if __name__ == "__main__":
    main()