# Plan de Implementación - Face Recognition Pipeline

> Plan incremental para construir el sistema paso a paso, probando en cada fase.

---

## Estado General
- **Iniciado**: 2025-11-04
- **Fase Actual**: Fase 1 - Configuración y Fundamentos
- **Última Actualización**: 2025-11-04

---

## FASE 0: Setup Inicial y Estructura Base ✅
**Objetivo**: Preparar el entorno y estructura del proyecto sin código complejo.

### Tareas:
- [x] Actualizar `pyproject.toml` con todas las dependencias
- [x] Instalar dependencias con `uv sync`
- [x] Crear estructura completa de directorios
- [x] Crear `.env.example` con variables de configuración
- [x] Crear archivos `__init__.py` vacíos en módulos

### Verificación:
```bash
# Verificar que uv instala correctamente
uv sync

# Verificar estructura de directorios
tree -L 2 .

# Verificar que Python importa los módulos
python -c "import app; print('OK')"
```

### Criterio de Éxito:
- ✅ Todas las dependencias instaladas sin errores
- ✅ Estructura de directorios completa
- ✅ Módulos Python importables

---

## FASE 1: Configuración y Fundamentos ⬜
**Objetivo**: Implementar configuración y tipos base sin lógica de ML.

### Tareas:
- [ ] Implementar `app/config.py` (cargar variables de .env)
- [ ] Implementar `app/logging_config.py` (configuración de logs)
- [ ] Implementar `app/interfaces.py` (Protocols, dataclasses)
- [ ] Crear test simple para config

### Verificación:
```bash
# Crear .env desde .env.example
cp .env.example .env

# Probar configuración
python -c "from app.config import Config; c = Config(); print(f'CTX_ID: {c.ctx_id}, THRESH: {c.thresh}')"

# Probar logging
python -c "from app.logging_config import setup_logging; logger = setup_logging('test'); logger.info('Test log')"

# Probar interfaces
python -c "from app.interfaces import BBox, Detection; b = BBox(0,0,100,100); print(b)"
```

### Criterio de Éxito:
- ✅ Config carga variables desde .env correctamente
- ✅ Logger funciona y muestra mensajes formateados
- ✅ Dataclasses se pueden instanciar

---

## FASE 2: Detector (Primera Funcionalidad Visual) ⬜
**Objetivo**: Detectar caras en imágenes y mostrar bounding boxes.

### Tareas:
- [ ] Implementar `app/detector_scrfd.py`
- [ ] Implementar `app/overlay.py` (dibujar boxes y landmarks)
- [ ] Crear script de prueba `scripts/test_detector.py`

### Verificación:
```bash
# Probar detector con webcam (debe dibujar boxes en caras detectadas)
python scripts/test_detector.py

# Verificar que detecta caras y extrae landmarks
# Debe mostrar ventana con:
# - Bounding boxes verdes alrededor de caras
# - 5 puntos de landmarks (ojos, nariz, boca)
# - Confidence score en pantalla
```

### Criterio de Éxito:
- ✅ Detector identifica caras en frames de webcam
- ✅ Se visualizan bounding boxes y landmarks correctamente
- ✅ Confidence scores son razonables (>0.5 para caras claras)

---

## FASE 3: Aligner (Normalización de Caras) ⬜
**Objetivo**: Alinear caras detectadas a formato estándar 112x112.

### Tareas:
- [ ] Implementar `app/aligner_fivept.py`
- [ ] Implementar `app/utils.py` (quality checks: sharpness, etc.)
- [ ] Crear script de prueba `scripts/test_aligner.py`

### Verificación:
```bash
# Probar alineación (debe mostrar caras alineadas y normalizadas)
python scripts/test_aligner.py

# Verificar que:
# - Caras detectadas se alinean a 112x112
# - Rostros están centrados y con orientación correcta
# - Se filtran imágenes borrosas (bajo Laplacian)
# - Se guardan crops alineados en data/test_crops/
```

### Criterio de Éxito:
- ✅ Caras alineadas tienen tamaño 112x112
- ✅ Orientación facial es consistente
- ✅ Filtro de calidad rechaza imágenes borrosas

---

## FASE 4: Embedder (Extracción de Features) ⬜
**Objetivo**: Convertir caras alineadas en vectores de 512 dimensiones.

### Tareas:
- [ ] Implementar `app/embedder_arcface.py`
- [ ] Crear script de prueba `scripts/test_embedder.py`

### Verificación:
```bash
# Probar embedder con crops alineados
python scripts/test_embedder.py

# Verificar que:
# - Embeddings tienen shape (512,)
# - Vectores están normalizados (L2 norm ≈ 1.0)
# - Misma cara → embeddings similares (cosine sim > 0.9)
# - Caras diferentes → embeddings diferentes (cosine sim < 0.5)
```

### Criterio de Éxito:
- ✅ Embeddings tienen dimensión correcta (512)
- ✅ Vectores normalizados (norm ≈ 1.0)
- ✅ Similitud intra-persona > 0.9
- ✅ Similitud inter-persona < 0.5

---

## FASE 5: Matcher (Búsqueda con FAISS) ⬜
**Objetivo**: Buscar identidades usando índice FAISS.

### Tareas:
- [ ] Implementar `app/matcher_faiss.py`
- [ ] Crear `tests/test_matcher.py`
- [ ] Crear script de prueba `scripts/test_matcher.py`

### Verificación:
```bash
# Correr tests unitarios
pytest tests/test_matcher.py -v

# Probar búsqueda con datos sintéticos
python scripts/test_matcher.py

# Verificar que:
# - Index se construye correctamente
# - Búsqueda retorna labels correctos
# - Scores reflejan similitud (0-1 range)
# - Top-k funciona correctamente
```

### Criterio de Éxito:
- ✅ Tests unitarios pasan
- ✅ Búsqueda retorna identidad correcta para queries conocidas
- ✅ Scores son consistentes con similitud coseno

---

## FASE 6: Enrollment (Registro de Personas) ⬜
**Objetivo**: Sistema completo para enrollar nuevas personas.

### Tareas:
- [ ] Implementar `app/enrollment.py`
- [ ] Implementar `app/video_io.py` (WebcamSource, VideoFileSource)
- [ ] Implementar `scripts/capture_enroll.py`
- [ ] Implementar `scripts/build_index.py`

### Verificación:
```bash
# 1. Capturar imágenes de una persona
python scripts/capture_enroll.py --name LUIS --num-images 15

# Verificar que:
# - Se crean 15 imágenes en data/enroll/LUIS/
# - Imágenes son 112x112 alineadas
# - Solo se capturan imágenes con buena calidad

# 2. Construir índice FAISS
python scripts/build_index.py

# Verificar que:
# - Se crea models/centroids.faiss
# - Se crea models/labels.json con ["LUIS"]
# - Se crea models/stats.pkl
# - Logs muestran embeddings promediados
```

### Criterio de Éxito:
- ✅ Captura guiada funciona correctamente
- ✅ Se filtran imágenes de baja calidad
- ✅ Índice FAISS se construye con centroids
- ✅ Archivos de modelo se persisten correctamente

---

## FASE 7: Recognition (Identificación en Tiempo Real) ⬜
**Objetivo**: Sistema completo de reconocimiento en video.

### Tareas:
- [ ] Implementar `app/recognition.py` (servicio de reconocimiento)
- [ ] Implementar `scripts/run_webcam.py`
- [ ] Implementar `scripts/run_video.py`
- [ ] Crear `tests/test_enrollment.py`

### Verificación:
```bash
# 1. Reconocimiento en webcam
python scripts/run_webcam.py

# Verificar que:
# - Detecta caras en tiempo real
# - Muestra "LUIS" cuando aparece Luis
# - Muestra "unknown" para personas no enrolladas
# - Score de similitud se muestra junto al nombre
# - FPS se imprime en consola

# 2. Reconocimiento en video
python scripts/run_video.py --video path/to/video.mp4

# Verificar que:
# - Procesa video frame por frame
# - Identifica personas correctamente
# - Se puede pausar/reanudar con tecla
```

### Criterio de Éxito:
- ✅ Reconocimiento funciona en tiempo real (>10 FPS)
- ✅ Identifica correctamente personas enrolladas
- ✅ Threshold funciona (ajustar en .env)
- ✅ Overlay es claro y legible

---

## FASE 8: Tests y Calidad de Código ⬜
**Objetivo**: Asegurar calidad y mantenibilidad del código.

### Tareas:
- [ ] Crear `tests/test_embeds.py`
- [ ] Configurar ruff en pyproject.toml
- [ ] Configurar black en pyproject.toml
- [ ] Correr linters y formatters
- [ ] Agregar type hints faltantes

### Verificación:
```bash
# Correr todos los tests
pytest tests/ -v --cov=app

# Linting
ruff check .

# Formatting
black --check .

# Type checking (opcional)
mypy app/
```

### Criterio de Éxito:
- ✅ Cobertura de tests >70%
- ✅ No hay errores de ruff
- ✅ Código formateado con black
- ✅ Type hints en todas las funciones públicas

---

## FASE 9: Documentación Final ⬜
**Objetivo**: Documentar uso y deployment del sistema.

### Tareas:
- [ ] Escribir README.md completo
- [ ] Agregar docstrings a todas las clases públicas
- [ ] Crear guía de troubleshooting
- [ ] Agregar ejemplos de uso

### Verificación:
```bash
# Verificar que README tiene:
# - Sección de instalación
# - Ejemplos de uso para cada script
# - Troubleshooting común
# - Screenshots o GIFs demostrativos

# Verificar docstrings
python -c "from app.detector_scrfd import SCRFDDetector; help(SCRFDDetector)"
```

### Criterio de Éxito:
- ✅ README está completo y claro
- ✅ Todas las clases públicas tienen docstrings
- ✅ Hay ejemplos de uso funcionando

---

## Notas de Implementación

### Principios a seguir:
1. **Incremental**: Cada fase debe funcionar antes de avanzar
2. **Testeable**: Siempre crear scripts de prueba
3. **Visual**: Cuando sea posible, mostrar resultados visualmente
4. **Modular**: Mantener componentes desacoplados
5. **Documentado**: Comentar decisiones importantes

### Comandos útiles durante desarrollo:
```bash
# Correr script actual
python scripts/<script_name>.py

# Verificar imports
python -c "from app import <module>; print('OK')"

# Ver logs en tiempo real
tail -f logs/app.log  # (si implementamos file logging)

# Limpiar data de prueba
rm -rf data/enroll/TEST_*
rm -rf data/test_crops/
```

### Checklist rápido antes de cada commit:
- [ ] El código corre sin errores
- [ ] Los tests pasan
- [ ] Se agregaron docstrings
- [ ] Se probó manualmente la funcionalidad

---

## Historial de Cambios

### 2025-11-04
- ✅ Creado plan de implementación incremental
- ✅ FASE 0 completada: Setup inicial y estructura base
  - Dependencias instaladas (OpenCV, InsightFace, FAISS, etc.)
  - Estructura de directorios creada (app/, scripts/, tests/, data/, models/)
  - Configuración pyproject.toml con ruff, black, pytest
  - Archivo .env.example creado
- Estado actual: FASE 1 - Configuración y Fundamentos

---

## Próximos Pasos (Después de completar todas las fases)

### Features futuras (no implementar ahora):
- [ ] Attendance tracking con cooldown
- [ ] Liveness detection (anti-spoofing)
- [ ] FastAPI service
- [ ] Multi-camera support
- [ ] GPU optimization
- [ ] ONNX/TensorRT deployment
- [ ] Database persistence
- [ ] Web dashboard

### Mejoras de rendimiento:
- [ ] Batch processing
- [ ] Frame skipping inteligente
- [ ] Async video processing
- [ ] Model quantization

---

**¿Listo para empezar?**
Comencemos con la FASE 0: Setup Inicial y Estructura Base.