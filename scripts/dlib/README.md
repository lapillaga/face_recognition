# dlib/face_recognition - Pipeline de Reconocimiento Facial

Este pipeline usa **dlib** via la librería `face_recognition` (enfoque PyImageSearch):
- **Detector**: HOG (rápido) o CNN (más preciso)
- **Embedder**: ResNet-34 (genera embeddings de 128 dimensiones)
- **Matcher**: Distancia Euclidiana

## Flujo Visual

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE DLIB                                   │
└─────────────────────────────────────────────────────────────────────────┘

PASO 1: Codificación de Caras (TODO EN UN PASO)
┌─────────────────────────────────────────────────────────────────────────┐
│  Lee data/dataset/PERSONA/*.jpg                                         │
│       │                                                                 │
│       ▼                                                                 │
│  face_recognition.face_encodings()                                      │
│  (internamente: detecta → alinea → genera embedding 128-D)             │
│       │                                                                 │
│       ▼                                                                 │
│  Salida: models/encodings.pkl (embeddings + nombres)                   │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PASO 2: Reconocimiento
┌─────────────────────────────────────────────────────────────────────────┐
│  Webcam/Video/Imagen → HOG/CNN → face_encodings() → Distancia L2       │
│                                                                         │
│  Resultado: Nombre de la persona + distancia (menor = mejor)           │
└─────────────────────────────────────────────────────────────────────────┘
```

## Diferencia Principal con InsightFace

| Aspecto | InsightFace | dlib |
|---------|-------------|------|
| **Pasos** | 3 (captura → build → run) | 2 (encode → run) |
| **Input** | Webcam (captura) | Fotos existentes |
| **Guarda imágenes** | Sí (112x112) | No |
| **Proceso** | Modular (componentes separados) | Integrado (todo junto) |

## Scripts Disponibles

| Script | Descripción |
|--------|-------------|
| `01_encode_faces.py` | Procesa fotos y genera encodings |
| `02_run_webcam.py` | Reconocimiento en tiempo real via webcam |
| `02_run_video.py` | Reconocimiento en archivo de video |
| `02_run_image.py` | Reconocimiento en imagen estática |

## Pasos de Ejecución

### Preparación: Organiza tus fotos

Primero necesitas un dataset de fotos organizadas por persona:

```
data/
└── dataset/
    ├── LUIS/
    │   ├── foto1.jpg
    │   ├── foto2.jpg
    │   └── foto3.jpg
    ├── MARIA/
    │   ├── imagen1.png
    │   └── imagen2.jpg
    └── PEDRO/
        └── perfil.jpg
```

**Requisitos de las fotos:**
- Que se vea la cara claramente (frontal preferible)
- Buena iluminación
- Mínimo 3-5 fotos por persona
- Formatos: JPG, JPEG, PNG, BMP

### PASO 1: Codificar las caras

Procesa todas las fotos y extrae embeddings:

```bash
python scripts/dlib/01_encode_faces.py --dataset data/dataset
```

**Opciones:**
- `--dataset PATH`: Ruta al dataset de fotos (default: `data/dataset`)
- `--detection-method hog|cnn`: Método de detección (default: `hog`)
  - `hog`: Más rápido, menos preciso
  - `cnn`: Más lento, más preciso (requiere GPU)

**Salida:**
- `models/encodings.pkl` - Embeddings + nombres
- `models/labels_dlib.json` - Lista de nombres

### PASO 2: Ejecutar reconocimiento

#### Opción A: Webcam en tiempo real
```bash
python scripts/dlib/02_run_webcam.py
```

#### Opción B: Archivo de video
```bash
python scripts/dlib/02_run_video.py --video path/to/video.mp4
```

#### Opción C: Imagen estática
```bash
python scripts/dlib/02_run_image.py --image path/to/foto.jpg
```

**Opciones comunes:**
- `--detector-model hog|cnn`: Método de detección (default: hog)
- `--threshold 0.4`: Umbral de similitud (0-1, mayor = más estricto)
- `--no-display`: Ejecutar sin mostrar ventana

## Ejemplo Completo

```bash
# 1. Organiza tus fotos en data/dataset/PERSONA/*.jpg

# 2. Codificar caras
python scripts/dlib/01_encode_faces.py --dataset data/dataset

# 3. Probar reconocimiento
python scripts/dlib/02_run_webcam.py
```

## Estructura de Archivos

```
data/
└── dataset/           # TUS FOTOS (input)
    ├── LUIS/
    │   ├── foto1.jpg
    │   └── foto2.jpg
    └── MARIA/
        └── foto1.jpg

models/
├── encodings.pkl      # Embeddings (128-D) + nombres
└── labels_dlib.json   # ["LUIS", "MARIA", ...]
```

## Ajuste del Threshold

El threshold representa **similitud** (0 a 1):
- **Mayor similitud = más parecido**
- Se calcula como: `similitud = 1 - distancia`
- Default: **0.4** (equivale a distancia máxima de 0.6)

| Threshold | Comportamiento |
|-----------|----------------|
| 0.3 | Permisivo (más matches, más falsos positivos) |
| 0.4 | Balance recomendado |
| 0.5 | Estricto (menos matches) |
| 0.6 | Muy estricto (pocos matches) |

```bash
# Más permisivo (acepta matches con menor similitud)
python scripts/dlib/02_run_webcam.py --threshold 0.3

# Más estricto (requiere mayor similitud)
python scripts/dlib/02_run_webcam.py --threshold 0.5
```

## Nota sobre Rendimiento

- **HOG**: ~5-10 FPS en CPU
- **CNN**: ~1-3 FPS en CPU, ~15-30 FPS en GPU

Para mejor rendimiento:
```bash
# Usar HOG (más rápido)
python scripts/dlib/01_encode_faces.py --detection-method hog
python scripts/dlib/02_run_webcam.py --detector-model hog
```