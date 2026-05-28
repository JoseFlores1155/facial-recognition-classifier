import cv2
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
    
    # Guardamos el modelo resultante
    face_recognizer.write("modelo_lbph.xml")
    print("¡Modelo entrenado exitosamente y guardado como 'modelo_lbph.xml'!")
else:
    print("No se pudieron capturar rostros suficientes.")
