import cv2

# Lista de usuarios indexada por el ID (el ID 0 corresponde a la primera posición)
NOMBRES = ["Jose Flores"] 

# Cargamos el clasificador de detección y el modelo entrenado
cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(cascade_path)

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
try:
    face_recognizer.read("modelo_lbph.xml")
except cv2.error:
    print("Error: No se encontró el archivo 'modelo_lbph.xml'. Corre primero entrenar.py")
    exit()

cap = cv2.VideoCapture(0)

print("== Reconocimiento activado ==")
print("Presiona 'q' para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rostros = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in rostros:
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(roi_gray, (150, 150), interpolation=cv2.INTER_CUBIC)

        # El método predict nos devuelve el ID y la distancia/confianza
        id_predicho, confianza = face_recognizer.predict(roi_gray)

        # En LBPH, una "confianza" menor a 70 u 80 suele ser una buena coincidencia
        # (A menor valor, mayor similitud matemática)
        if confianza < 75:
            nombre = NOMBRES[id_predicho]
            color = (0, 255, 0) # Verde si lo reconoce
            texto = f"{nombre} ({int(confianza)})"
        else:
            color = (0, 0, 255) # Rojo si es desconocido
            texto = "Desconocido"

        # Dibujar UI en el video
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, texto, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("Sistema de Reconocimiento Facial", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Sistema apagado.")
