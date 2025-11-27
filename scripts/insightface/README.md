# InsightFace - Pipeline de Reconocimiento Facial

Este pipeline usa **InsightFace** con los siguientes componentes:
- **Detector**: SCRFD (detecta caras + 5 landmarks faciales)
- **Aligner**: FivePointAligner (transforma a 112x112 usando landmarks)
- **Embedder**: ArcFace (genera embeddings de 512 dimensiones)
- **Matcher**: FAISS (búsqueda por similitud coseno)

## Flujo Visual

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PIPELINE INSIGHTFACE                            │
└─────────────────────────────────────────────────────────────────────────┘

PASO 1: Preparar imágenes (ELIGE UNA OPCIÓN)
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  OPCIÓN A: Webcam (01_capture_enroll.py)                               │
│  Webcam → SCRFD → Alinear → 112x112 → Guardar                          │
│                                                                         │
│  OPCIÓN B: Fotos existentes (01_process_dataset.py)                    │
│  Fotos cuerpo entero → SCRFD → Alinear → 112x112 → Guardar             │
│                                                                         │
│  Salida: data/enroll/PERSONA/*.jpg (imágenes alineadas)                │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PASO 2: Construcción del Índice (02_build_index.py)
┌─────────────────────────────────────────────────────────────────────────┐
│  Lee data/enroll/ → ArcFace (512-D) → Centroides → FAISS Index         │
│                                                                         │
│  Salida: models/centroids.faiss, models/labels.json                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
PASO 3: Reconocimiento (ELIGE UNA OPCIÓN)
┌─────────────────────────────────────────────────────────────────────────┐
│  03_run_webcam.py  → Webcam en tiempo real                             │
│  03_run_video.py   → Archivo de video                                  │
│  03_run_image.py   → Imagen estática                                   │
│                                                                         │
│  Resultado: Nombre de la persona + score de similitud                  │
└─────────────────────────────────────────────────────────────────────────┘
```

## Scripts Disponibles

| Script | Descripción |
|--------|-------------|
| `01_capture_enroll.py` | **Opción A**: Captura imágenes via webcam |
| `01_process_dataset.py` | **Opción B**: Procesa fotos existentes (cuerpo entero → cara) |
| `02_build_index.py` | Construye el índice FAISS a partir de las imágenes |
| `03_run_webcam.py` | Reconocimiento en tiempo real via webcam |
| `03_run_video.py` | Reconocimiento en archivo de video |
| `03_run_image.py` | Reconocimiento en imagen estática |

## Pasos de Ejecución

### PASO 1: Preparar imágenes de enrolamiento

Tienes **dos opciones** para obtener las imágenes 112x112:

#### OPCIÓN A: Captura via webcam (`01_capture_enroll.py`)

Usa la webcam para capturar imágenes en vivo:

```bash
python scripts/insightface/01_capture_enroll.py --name LUIS
python scripts/insightface/01_capture_enroll.py --name MARIA --num-images 20
```

**Opciones:**
- `--name NOMBRE`: Nombre de la persona (requerido)
- `--num-images 15`: Número de imágenes a capturar (default: 15)
- `--camera 0`: ID de la cámara (default: 0)

---

#### OPCIÓN B: Procesar fotos existentes (`01_process_dataset.py`)

Si ya tienes fotos (de cuerpo entero, selfies, etc.), este script detecta y recorta las caras automáticamente:

```bash
# Primero organiza tus fotos:
# data/my_dataset/LUIS/foto1.jpg, foto2.jpg, ...
# data/my_dataset/MARIA/foto1.jpg, ...

python scripts/insightface/01_process_dataset.py --dataset-dir data/my_dataset
```

**Opciones:**
- `--dataset-dir PATH`: Carpeta con fotos organizadas por persona
- `--min-sharpness 150`: Filtro de calidad (menor = más permisivo)
- `--replace`: Reemplazar imágenes existentes

**Ejemplo de estructura de entrada:**
```
data/my_dataset/
├── LUIS/
│   ├── foto_cuerpo_entero.jpg   # El script detecta la cara
│   ├── selfie.jpg
│   └── foto_grupal.jpg          # Extrae solo la cara principal
└── MARIA/
    ├── perfil.png
    └── foto.jpg
```

---

**Salida de ambas opciones:** `data/enroll/PERSONA/*.jpg` (112x112 alineadas)

### PASO 2: Construir el índice FAISS

Procesa todas las imágenes y crea el índice de búsqueda:

```bash
python scripts/insightface/02_build_index.py
```

**Salida:**
- `models/centroids.faiss` - Índice FAISS
- `models/labels.json` - Lista de nombres
- `models/stats.pkl` - Estadísticas

### PASO 3: Ejecutar reconocimiento

#### Opción A: Webcam en tiempo real
```bash
python scripts/insightface/03_run_webcam.py
```

#### Opción B: Archivo de video
```bash
python scripts/insightface/03_run_video.py --video path/to/video.mp4
```

#### Opción C: Imagen estática
```bash
python scripts/insightface/03_run_image.py --image path/to/foto.jpg
```

**Opciones comunes:**
- `--threshold 0.35`: Umbral de similitud (0.0 a 1.0)
- `--no-display`: Ejecutar sin mostrar ventana

## Ejemplo Completo

```bash
# 1. Enrolar personas
python scripts/insightface/01_capture_enroll.py --name LUIS --num-images 15
python scripts/insightface/01_capture_enroll.py --name MARIA --num-images 15

# 2. Construir índice
python scripts/insightface/02_build_index.py

# 3. Probar reconocimiento
python scripts/insightface/03_run_webcam.py --threshold 0.4
```

## Estructura de Archivos Generados

```
data/
└── enroll/
    ├── LUIS/
    │   ├── 000.jpg    # 112x112 alineada
    │   ├── 001.jpg
    │   └── ...
    └── MARIA/
        ├── 000.jpg
        └── ...

models/
├── centroids.faiss    # Índice FAISS
├── labels.json        # ["LUIS", "MARIA", ...]
└── stats.pkl          # Estadísticas
```

## Ajuste del Threshold

El threshold controla cuán estricto es el reconocimiento:

| Threshold | Comportamiento |
|-----------|----------------|
| 0.25 | Muy permisivo (más falsos positivos) |
| 0.35 | Balance recomendado |
| 0.45 | Estricto (más "unknown") |
| 0.55 | Muy estricto |

Ajusta según tus necesidades:
```bash
python scripts/insightface/03_run_webcam.py --threshold 0.40
```