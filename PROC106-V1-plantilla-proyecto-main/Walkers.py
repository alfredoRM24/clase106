import cv2


# Crear nuestro clasificador de cuerpos
clasificador=cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Inicializar la captura de video para nuestro archivo de video
cap = cv2.VideoCapture('walking.avi')

# Comenzar el bucle una vez que el video esté cargado exitosamente
while True:
    
    # Leer el primer cuadro
    ret, frame = cap.read()

    # Convertir cada cuadro a escala de grises
    cuadros = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    # Pasar el cuadro a nuestro clasificador de cuerpos
    bodie = clasificador.detectMultiScale(cuadros,1.1,5)
    
    # Extraer las cajas envolventes para cualquier cuerpo identificado
    for (x,y,w,h) in bodie:
        cv2.rectangle(frame,(x,y),(x+w, y+h),(129,0,147),2)

    cv2.imshow("detectado",frame)
    if cv2.waitKey(1) == 32: #32 es la barra espaciadora
        break

cap.release()
cv2.destroyAllWindows()
