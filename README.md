# Reconhecimento de Emoções em Faces

Projeto final da disciplina **Práticas em Ciência de Dados III (PCD3)** — ICMC-USP, 2026.

**Integrantes:**
- Gustavo Cardozo De Moraes Moreira — NUSP 5244057
- Francisco Luiz Maian do Nascimento — NUSP 14570890

---

## Visão Geral

O projeto aborda o problema de **Facial Expression Recognition (FER)** utilizando o dataset FER2013. Dois modelos são comparados: uma CNN compacta treinada do zero (baseline) e uma ResNet-18 pré-treinada fine-tuned em duas fases. O melhor modelo é integrado a uma aplicação de inferência em tempo real via webcam.

---

## Estrutura do Repositório

```
.
├── main.ipynb      # Pipeline completo: dados, treino, avaliação e comparação
├── realtime_emotion.py            # Inferência em tempo real via webcam (OpenCV)
├── baseline_cnn.pth               # Pesos salvos do modelo CNN baseline
├── resnet18_fer_finetuned.pth     # Pesos do melhor modelo ResNet-18 fine-tuned
└── README.md
```

---

## Dependências

- Python 3.x
- PyTorch + torchvision
- OpenCV (`cv2`)
- NumPy
- Matplotlib
- scikit-learn

Instale todas com:

```bash
pip install torch torchvision opencv-python numpy matplotlib scikit-learn
```

---

## Dataset — FER2013

| Partição | Amostras | Formato | Canais |
|---|---|---|---|
| Treino | 28.709 | 48×48 px | Escala de cinza |
| Validação / Teste | 7.178 | 48×48 px | Escala de cinza |
| **Total** | **35.887** | | |

As 7 classes são: **Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral**.

O dataset apresenta forte desbalanceamento: a classe *Happy* possui cerca de 3,5× mais amostras que *Disgust*, a menor. Para mitigar esse efeito, foi adotada a estratégia de **pesos de classe** na função de perda `CrossEntropyLoss`, com peso inversamente proporcional à frequência de cada classe:

```
w_c = N / (K × n_c)
```

---

## Pipeline

```
FER2013 → Normalização → Data Augmentation → Pesos de classe
              ↓                                      ↓
        CNN Baseline                         ResNet-18 Fine-tuning
    (4 blocos conv, do zero)             (bifásico, LR diferenciado)
              ↓                                      ↓
         Avaliação (Acc, F1-macro, Precisão, Recall, Conf. Matrix)
              ↓
     Aplicação em Tempo Real (Webcam → Haar Cascade → Inferência → Display)
```

---

## Modelos

### CNN Baseline

CNN compacta com 4 blocos convolucionais (Conv → ReLU → MaxPool), filtros crescentes (32, 64, 128, 256), seguidos de classificador denso (2304 → 256 → 7) com Dropout de 50%.

| Hiperparâmetro | Valor |
|---|---|
| Otimizador | Adam (lr = 1e-3) |
| Scheduler | StepLR (step=15, γ=0.5) |
| Épocas | 30 |
| Batch size | 64 |

### ResNet-18 Fine-tuned

ResNet-18 pré-treinada no ImageNet com duas adaptações de entrada (replicação do canal único para RGB falso + redimensionamento 48→224 px) e head personalizado (512→256→7) com Dropout de 40%.

O fine-tuning é realizado em duas fases:

**Fase 1 — Extração de features (backbone congelado)**

| Parâmetro | Valor |
|---|---|
| Épocas | 10 |
| Otimizador | Adam (lr = 1e-3) |
| Parâmetros treináveis | ~132K (apenas head) |

**Fase 2 — Fine-tuning completo**

| Parâmetro | Valor |
|---|---|
| Épocas | 30 |
| LR backbone | 1e-5 |
| LR head | 1e-4 |
| Scheduler | Cosine Annealing (Tmax=30) |
| Parâmetros treináveis | ~11M (todos) |

### Data Augmentation

Ambos os modelos usam: flip horizontal aleatório, rotação ±15°, RandomResizedCrop (90–110%). A ResNet-18 inclui também ColorJitter (brilho e contraste 0.3) para maior robustez a variações de iluminação.

---

## Resultados

| Modelo | Acurácia (%) | F1-macro | Precisão-macro | Recall-macro |
|---|---|---|---|---|
| CNN Baseline | 57.44 | 0.5415 | 0.5333 | 0.5769 |
| ResNet-18 Fine-tuned | **64.60** | **0.6238** | **0.6172** | **0.6392** |

A ResNet-18 supera a baseline em todas as métricas. As classes mais difíceis são **Fear** e **Disgust**, frequentemente confundidas com Sad e Angry, respectivamente. **Happy** é a classe mais fácil de classificar em ambos os modelos.

---

## Aplicação em Tempo Real — `realtime_emotion.py`

A cada 3 quadros capturados pela webcam, o detector **Haar Cascade** (OpenCV) localiza a face na cena. A região de interesse passa pelo mesmo pré-processamento do treino (escala de cinza → 224×224 → RGB falso → normalização ImageNet) e é classificada pela ResNet-18. A emoção predita é exibida como bounding box colorido com barra de confiança.

```bash
python realtime_emotion.py
```

---
