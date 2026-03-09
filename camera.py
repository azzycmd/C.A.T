import cv2
import mediapipe as mp
import math
import numpy as np

# os coiso la do mediapipe
mp_hands = mp.solutions.hands
mp_face = mp.solutions.face_mesh
hands = mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_mesh = mp_face.FaceMesh(max_num_faces=1)
mp_drawing = mp.solutions.drawing_utils

# toggle
censor = False
censorFace = False

# titulo camera
tituloCamera = "wow, camera"

# cores
HAND_COLOR = (255, 255, 255) 
FACE_COLOR = (255, 255, 255)  
TXT_COLOR = (0, 255, 0)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success: 
        continue

    # bgr --> rgb
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # as moes
    results_hands = hands.process(image_rgb)
    if results_hands.multi_hand_landmarks:
        for hand_landmarks in results_hands.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                       mp_drawing.DrawingSpec(color=HAND_COLOR, thickness=2, circle_radius=2))
            landmarks = hand_landmarks.landmark
            
            h, w, _ = image.shape
            
            # prep moes todas
            ponta_indicador = landmarks[6].y
            base_indicador = landmarks[8].y
            algo_indicador = landmarks[7].y

            ponta_meio = landmarks[12].y
            base_meio = landmarks[10].y

            # prep coiso v0.25
            tips_ids = [4, 8, 12, 16, 20]

            #prep pinca
            x8 = landmarks[8].x
            y8 = landmarks[8].y
            x4 = landmarks[4].x
            y4 = landmarks[4].y

            # prep coiso v0.5
            abertos = []
            # prep coiso v1
            if landmarks[0].y < landmarks[tips_ids[0] - 1].y:
                abertos.append(True)
            else:
                abertos.append(False)

            dist = math.hypot(x8 - x4, y8 - y4)

            # prep coiso v2
            for i in range(1, 5):
                if landmarks[tips_ids[i]].y < landmarks[tips_ids[i] - 2].y:
                    abertos.append(True)
                else:
                    abertos.append(False)

            # moes quadrdo
            if dist < 0.04 and landmarks[20].y < landmarks[18].y: # pinca
                TXT_COLOR = (255, 255, 0)
                texto = "pinca"
            elif abertos[0] == True and all(f == False for f in abertos[1:]):
                TXT_COLOR = (255, 0, 255) # coiso
                texto = "coiso"    
            elif ponta_indicador < base_indicador and ponta_indicador < algo_indicador:
                TXT_COLOR = (0, 0, 255) # fachada
                texto = "fechada"
            else:
                TXT_COLOR = (0, 255, 0) # levantad0o
                texto = "levantado"

            #quadrado
            x_min = int(min([lm.x for lm in landmarks]) * w)
            y_min = int(min([lm.y for lm in landmarks]) * h)
            x_max = int(max([lm.x for lm in landmarks]) * w)
            y_max = int(max([lm.y for lm in landmarks]) * h)

            # Verifica se apenas o dedo do meio (índice 2 da lista abertos) está levantado
            if len(abertos) > 2 and abertos[2] == True and abertos[1] == False and abertos[3] == False and censor == True:
                # Define a região da mão com uma pequena margem
                y1, y2 = max(0, y_min-5), min(h, y_max+10)
                x1, x2 = max(0, x_min-20), min(w, x_max+20)
                
                TXT_COLOR = (255, 255, 255) # fachada
                texto = ""
                
                roi = image[y1:y2, x1:x2]
                if roi.size > 0:
                    # 1. Diminui a imagem (ex: 15x15 pixels)
                    # 2. Aumenta de volta para o tamanho original com INTER_NEAREST
                    temp = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
                    pixelado = cv2.resize(temp, (x2-x1, y2-y1), interpolation=cv2.INTER_NEAREST)
                    
                    # Aplica o mosaico de volta na imagem principal
                    image[y1:y2, x1:x2] = pixelado

            cv2.rectangle(image, (x_min-20, y_min-5), (x_max+20, y_max+10), TXT_COLOR, 2)
            cv2.putText(image, texto, (x_min, y_min-10), cv2.FONT_ITALIC, 0.5, TXT_COLOR, 2)

    # quadrado cara
    results_face = face_mesh.process(image_rgb)
    if results_face.multi_face_landmarks:
        for face_landmarks in results_face.multi_face_landmarks:
            coords = [(lm.x, lm.y) for lm in face_landmarks.landmark]
            x_min_f = int(min(c[0] for c in coords) * image.shape[1])
            y_min_f = int(min(c[1] for c in coords) * image.shape[0])
            x_max_f = int(max(c[0] for c in coords) * image.shape[1])
            y_max_f = int(max(c[1] for c in coords) * image.shape[0])

            h, w, _ = image.shape

            if censorFace == True:
                # Define a região da mão com uma pequena margem
                y1f, y2f = max(0, y_min_f), min(h, y_max_f)
                x1f, x2f = max(0, x_min_f), min(w, x_max_f)
                
                TXT_COLOR = (255, 255, 255) # fachada
                texto = ""
                
                roi = image[y1f:y2f, x1f:x2f]
                if roi.size > 0:
                    # 1. Diminui a imagem (ex: 15x15 pixels)
                    # 2. Aumenta de volta para o tamanho original com INTER_NEAREST
                    tempf = cv2.resize(roi, (10, 10), interpolation=cv2.INTER_LINEAR)
                    pixeladof = cv2.resize(tempf, (x2f-x1f, y2f-y1f), interpolation=cv2.INTER_NEAREST)
                    
                    # Aplica o mosaico de volta na imagem principal
                    image[y1f:y2f, x1f:x2f] = pixeladof
            
            cv2.rectangle(image, (x_min_f, y_min_f), (x_max_f, y_max_f), FACE_COLOR, 2)
            cv2.putText(image, "cara", (x_min_f, y_min_f-10), cv2.FONT_ITALIC , 0.5, FACE_COLOR, 2)

    cv2.imshow(tituloCamera, image)
    if cv2.waitKey(5) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()
