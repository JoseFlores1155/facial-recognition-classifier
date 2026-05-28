[Detecta_rostros_y_entrena_el_modelo.py](https://github.com/user-attachments/files/28329819/Detecta_rostros_y_entrena_el_modelo.py)[README_facial_recognition.md](https://github.com/user-attachments/files/28329814/README_facial_recognition.md)

# Facial Recognition System 🎯

> Sistema de reconocimiento facial en tiempo real usando el algoritmo **LBPH (Local Binary Patterns Histograms)** de OpenCV. Detecta e identifica usuarios registrados a través de la cámara, con entrenamiento personalizado por usuario.

---

## ¿Cómo funciona?

El sistema opera en dos fases independientes:

### Fase 1 — Entrenamiento (`Detecta_rostros_y_entrena_el_modelo.py`)
1. Activa la cámara y detecta el rostro usando **Haar Cascade** (`haarcascade_frontalface_default.xml`)
2. Captura **30 muestras** del rostro, redimensionadas a 150×150 px para homogeneidad
3. Entrena un reconocedor **LBPH** con las muestras capturadas
4. Guarda el modelo entrenado en `modelo_lbph.xml`

### Fase 2 — Reconocimiento en tiempo real (`conocer.py`)
1. Carga el modelo entrenado (`modelo_lbph.xml`)
2. Detecta rostros frame a frame con Haar Cascade
3. Predice la identidad con LBPH y evalúa la **confianza** del resultado
4. Umbral de confianza: `< 75` → usuario reconocido (verde) | `≥ 75` → desconocido (rojo)
5. Muestra nombre y nivel de confianza superpuesto en el video en vivo

---

## Tecnologías

| Herramienta | Uso |
|---|---|
| Python 3.x | Lógica principal |
| OpenCV (`cv2`) | Captura de video, detección facial, reconocimiento LBPH |
| NumPy | Conversión de listas de imágenes a arrays para entrenamiento |
| `os` | Gestión de directorios para almacenamiento de datos |
| Haar Cascade | Detector de rostros preentrenado (frontal) |
| LBPH Face Recognizer | Algoritmo de reconocimiento facial local por patrones binarios |

---

## Estructura del proyecto

```
facial-recognition-system/
│
├── Detecta_rostros_y_entrena_el_modelo.py   # Fase 1: captura + entrenamiento
├── conocer.py                                # Fase 2: reconocimiento en tiempo real
│
├── rostros/                                  # Carpeta generada automáticamente
│                                             # (imágenes de entrenamiento)
├── modelo_lbph.xml                           # Modelo entrenado (generado en Fase 1)
│
└── README.md
```

---

## Cómo ejecutar

### Requisitos
```bash
pip install opencv-contrib-python numpy
```

> ⚠️ Usar `opencv-contrib-python` (no `opencv-python`) — incluye el módulo `cv2.face` necesario para LBPH.

### Paso 1 — Entrenar el modelo con tu rostro
```bash
python "Detecta_rostros_y_entrena_el_modelo.py"
```
- Mira directo a la cámara
- El sistema capturará 30 muestras automáticamente
- Al terminar genera `modelo_lbph.xml`

### Paso 2 — Activar reconocimiento en tiempo real
```bash
python conocer.py
```
- Presiona `q` para salir

---

## Lógica de confianza LBPH

En el algoritmo LBPH, la métrica de confianza funciona de forma **inversa** a lo intuitivo:

| Valor de confianza | Interpretación |
|---|---|
| `0 – 50` | Coincidencia muy alta |
| `50 – 75` | ✅ Reconocido (umbral configurado) |
| `> 75` | ❌ Desconocido |

El umbral de `75` puede ajustarse en `conocer.py` según las condiciones de iluminación del entorno.

---

## Parámetros clave configurables

```python
# En Detecta_rostros_y_entrena_el_modelo.py
NOMBRE_USUARIO = "Jose Flores"   # Nombre del usuario a registrar
count < 30                        # Número de muestras de entrenamiento

# En conocer.py
NOMBRES = ["Jose Flores"]         # Lista de usuarios por ID
confianza < 75                    # Umbral de reconocimiento (ajustable)
scaleFactor=1.3, minNeighbors=5  # Sensibilidad del detector Haar Cascade
```

---

## Casos de uso potenciales

- Control de acceso automatizado en plantas industriales
- Registro de asistencia sin contacto
- Seguridad perimetral con identificación de personal autorizado

---

## Resultado

- ✅ Detección facial en tiempo real con clasificación visual (verde/rojo)
- ✅ Modelo LBPH entrenado con datos propios desde cámara
- ✅ Sistema funcional con dos scripts independientes y bien definidos
- ✅ Umbral de confianza configurable según condiciones del entorno
[Uploading Detecta_rostros_y_entrimport cv2
import os
import numpy as np

# Configuración inicial
NOMBRE_USUARIO = "Jose Flores"  
CARPETA_DATA = "rostros"

if not os.path.exists(CARPETA_DATA):
    os.makedirs(CARPETA_DATA)

# Cargamos el clasificador preentrenado para detectar rostros
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

cap = cv2.VideoCapture(0)
count = 0

print("== Fase 1: Capturando imágenes para el entrenamiento ==")
print("Mira a la cámara y espera a que termine...")

rostros_data = []
ids = []

while count < 30:  # Tomaremos 30 muestras
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in rostros:
        roi_gray = gray[y:y+h, x:x+w]
        # Redimensionamos para homogeneidad
        roi_gray = cv2.resize(roi_gray, (150, 150), interpolation=cv2.INTER_CUBIC)
        
        rostros_data.append(roi_gray)
        ids.append(0)  # ID asignado a este usuario (0 para el primero)
        
        count += 1
        # Dibujamos un rectángulo en el frame para ver el progreso
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, f"Capturando: {count}/30", (x, y-10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    cv2.imshow("Registrando Rostro", frame)
    
    # Delay corto y opción de salir con 'q'
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if len(rostros_data) > 0:
    print("\n== Fase 2: Entrenando el modelo LBPH... ==")
    # Inicializamos el reconocedor de caras por LBPH
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Entrenamos pasándole la lista de imágenes y la lista de IDs (convertida a numpy array)
    face_recognizer.train(rostros_data, np.array(ids))
    opencv-contrib-python>=4.8.0
numpy>=1.24.0

    # Guardamos el modelo resultante
    face_recognizer.write("modelo_lbph.xml")
    print("¡Modelo entrenado exitosamente y guardado como 'modelo_lbph.xml'!")
else:
    print("No se pudieron capturar rostros suficientes.")
ena_el_modelo.py…]()

[conocer.py](https://github.com/user-attachments/files/28329825/conocer.py)
[requirements_facial.txt](https://github.com/user-attachments/files/28329826/requirements_facial.txt)
---

> **Nota:** Este proyecto es un prototipo de demostración local. No almacena ni transmite datos biométricos de terceros.
