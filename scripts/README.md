# Scripts de Reconocimiento Facial

Este proyecto implementa dos pipelines de reconocimiento facial:

## Estructura

```
scripts/
├── insightface/              # Pipeline InsightFace (producción)
│   ├── README.md             # Documentación detallada
│   ├── 01_capture_enroll.py  # Opción A: Captura via webcam
│   ├── 01_process_dataset.py # Opción B: Procesa fotos existentes
│   ├── 02_build_index.py     # Construye índice FAISS
│   ├── 03_run_webcam.py      # Reconocimiento: webcam
│   ├── 03_run_video.py       # Reconocimiento: video
│   └── 03_run_image.py       # Reconocimiento: imagen
│
├── dlib/                     # Pipeline dlib (tutorial/educativo)
│   ├── README.md             # Documentación detallada
│   ├── 01_encode_faces.py    # Codifica fotos existentes
│   ├── 02_run_webcam.py      # Reconocimiento: webcam
│   ├── 02_run_video.py       # Reconocimiento: video
│   └── 02_run_image.py       # Reconocimiento: imagen
│
└── tests/                    # Scripts de prueba
    ├── test_detector.py
    ├── test_aligner.py
    ├── test_embedder.py
    └── test_matcher.py
```

## Comparación de Pipelines

| Aspecto | InsightFace | dlib |
|---------|-------------|------|
| **Precisión** | ⭐⭐⭐⭐⭐ Alta | ⭐⭐⭐ Media |
| **Velocidad** | ⭐⭐⭐⭐ Rápido | ⭐⭐⭐ Medio |
| **Facilidad** | ⭐⭐⭐ Media | ⭐⭐⭐⭐⭐ Fácil |
| **Embedding** | 512-D | 128-D |
| **Matcher** | FAISS (coseno) | L2 (euclidiana) |
| **Uso ideal** | Producción | Tutorial/Educativo |

## Flujo Rápido

### InsightFace (3 pasos)

```bash
# Paso 1: Capturar caras via webcam
python scripts/insightface/01_capture_enroll.py --name LUIS

# Paso 2: Construir índice FAISS
python scripts/insightface/02_build_index.py

# Paso 3: Reconocer
python scripts/insightface/03_run_webcam.py
```

### dlib (2 pasos)

```bash
# Paso 1: Codificar fotos existentes
python scripts/dlib/01_encode_faces.py --dataset data/dataset

# Paso 2: Reconocer
python scripts/dlib/02_run_webcam.py --backend dlib
```

## ¿Cuál elegir?

- **InsightFace**: Si necesitas máxima precisión y rendimiento
- **dlib**: Si quieres entender el proceso o hacer un prototipo rápido

## Documentación Detallada

- [InsightFace README](insightface/README.md)
- [dlib README](dlib/README.md)
